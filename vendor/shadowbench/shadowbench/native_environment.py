"""V3.1 native industrial hand/forearm skin, geometric gates and explicit posture strain.

The current reviewed-book environment builds on this unchanged implementation, and
recorded v3.1 episodes replay through it. It never claims that an ergonomic posture
surrogate measures human pain.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import threading
import time

import fcl
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .mesh import Mesh,_inside,collision_object
from .pose import InvalidPose,Pose,SIDES,finite_array,interpolate
from .render import _raster,comparison,shadow,wall
from .score import grade
from .targets import Task
from .ultraleap import ASSET,NativeRig

LOG=logging.getLogger(__name__)
ALIASES=tuple(f"{finger}.{joint}" for finger in ("index","middle","ring","little")
              for joint in ("mcp","spread","pip","dip"))+(
    "thumb.base","thumb.opposition","thumb.mcp","thumb.ip","wrist.flex","wrist.side","forearm.roll")
INDEX={alias:i for i,alias in enumerate(ALIASES)}
MODEL_NAME="Ultraleap native hand + forearm"
BODY={"type":"object","additionalProperties":False}
VEC={"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3}
HAND={"type":"string","enum":["right","left"]}
SPECS={
 "describe":("Read native-rest-relative joint coordinates, conservative ranges, hand directions and explicit strain components.",{}),
 "look":("One image with four panels: fixed target, your shadow, overlay (red missing/blue extra/gray unknown), human-looking hand view. Costs one image.",{"view":{"type":"string","enum":["front","back","side","top","iso"]}}),
 "set_fingers":("Set absolute control coordinates in degrees relative to the native relaxed pose. Positive mcp/pip/dip curl fingers. Flat keys index.mcp/pip/dip/spread, etc., thumb.base/opposition/mcp/ip, wrist.flex/side, forearm.roll. Bounds and current values are in describe; unspecified joints stay unchanged.",{"hand":HAND,"joints":{"type":"object","additionalProperties":{"type":"number"}}}),
 "move_hand":("Translate one hand/forearm by a world-space delta in meters (at most 15cm per call). Light is fixed.",{"hand":HAND,"delta":VEC}),
 "turn_hand":("Rotate one whole hand/forearm around its wrist anchor, right-hand-rule axis in world coordinates, at most 45 degrees per call.",{"hand":HAND,"axis":VEC,"degrees":{"type":"number"}}),
 "undo":("Undo the last accepted action; refused actions did not change the pose.",{}),
 "submit":("Finish the episode against the immutable silhouette. Any admissible solution is acceptable.",{}),
}
TOOL_SPECS={name:(description,{**BODY,"properties":props,"required":list(props)})
            for name,(description,props) in SPECS.items()}
LIMITS={
    **{f"{finger}.{joint}":limits for finger in ("index","middle","ring","little")
       for joint,limits in (("mcp",(-5.,70.)),("spread",(-15.,15.)),("pip",(0.,95.)),("dip",(0.,70.)))},
    "thumb.base":(-25.,40.),"thumb.opposition":(-35.,35.),"thumb.mcp":(-5.,55.),"thumb.ip":(-5.,70.),
    "wrist.flex":(-50.,50.),"wrist.side":(-25.,15.),"forearm.roll":(-80.,80.),
}
SYSTEM_PROMPT="""Reproduce the supplied fixed silhouette with two human-shaped 3D hands using only the tools.
The skin AND forearms come from Ultraleap's native rig; no other skeleton is substituted.
World meters: x right, y up, z toward the fixed lamp [0,0,2.5]; screen z=0.
The fixed observed aperture is x [-0.4,0.4], y [-0.3,0.3]. Outside it is out of view.
Moving toward the lamp enlarges shadows. The target never moves or changes to match you.
Gray pixels are fixed uncertainty in the original artwork; other background and forearm shadows count.
Angles are absolute control coordinates in degrees RELATIVE TO THE ASSET'S RELAXED REST POSE.
Fingers mcp/pip/dip: positive curls toward the palm. Spread changes MCP abduction.
Thumb base/opposition are two native CMC axes; thumb mcp/ip are hinge controls.
Wrist flex/side bend the continuous wrist; positive side bends toward the thumb, negative toward the little finger.
Forearm.roll rotates the forearm longitudinally.
Describe gives current coordinates, permitted ranges, fingertip directions and strain components.
Hard gates enforce conservative range limits, decreasing available MCP spread during flexion,
actual skin triangle intersections, containment, stage bounds and intermediate motion checks.
The same continuous skin is rendered, casts the shadow and is collision checked.
These are model-bounded numerical checks, not full-body dynamics or clinical certification.
Refused actions cost a call and change nothing. No silently corrected angles or teleports.
Each set_fingers call can change several joints together, at most 45 degrees per joint.
Translations are at most 15cm and root rotations 45 degrees per call; smaller steps aid inspection.
Start with fingers up, palms toward the screen. To use one hand, move the other out of view gradually.
Look returns target, shadow, difference (red missing / blue extra), and colour-coded hand close-ups.
View options: front, back, side, top, iso. Inspect frequently after coherent changes.
Any admissible pose is acceptable; there is no canonical-pose imitation reward.
Reward is silhouette IoU, boundary detail and holes, with only 5% secondary posture ease.
Ease uses explicit joint-end-range, wrist-posture and soft finger-coupling costs. It is NOT measured pain,
muscle activation or a clinical risk score. The empirical mechanisms and engineering constants are disclosed.
100 calls, 32 look images, plus the initial observation. Submit when finished."""


@dataclass
class Output:
    text: str
    image: Image.Image | None = None


def state_hash(pose:Pose)->str:
    return hashlib.sha256(json.dumps(pose.dump(),sort_keys=True,separators=(",",":")).encode()).hexdigest()


def angles_for(pose:Pose,side:int)->dict[str,float]:
    return dict(zip(ALIASES,np.degrees(pose.q.reshape(2,-1)[side]).tolist()))


# Recorded v3.1 episodes carry the fingerprint of this implementation before its pose and mesh
# helpers moved, unchanged, into pose.py and mesh.py. replay_aliases.json maps exactly one
# relocated code state to it; scripts/audit_replay_v3.py re-simulated every recorded episode
# under that state (states, decisions, scores, observations). Any later edit changes the
# computed fingerprint, so the alias stops applying and those episodes are reported as retired.
REPLAY_ALIASES=Path(__file__).with_name("replay_aliases.json")


def computed_fingerprint()->str:
    sha=hashlib.sha256(b"shadowbench-v3.1-native-ultraleap")
    for name in ("native_environment.py","ultraleap.py","render.py","score.py","pose.py","mesh.py"):
        sha.update(Path(__file__).with_name(name).read_bytes())
    sha.update(ASSET.read_bytes())
    for package in ("numpy","scipy","pillow","opencv-python-headless","python-fcl"):
        sha.update(f"{package}={version(package)}".encode())
    return sha.hexdigest()


@lru_cache(maxsize=1)
def fingerprint()->str:
    computed=computed_fingerprint()
    aliases=json.loads(REPLAY_ALIASES.read_text()) if REPLAY_ALIASES.is_file() else {}
    return aliases.get(computed,computed)


def spread_limit(flexion:float)->float:
    return 3.+12.*np.cos(np.pi/2*np.clip(flexion/70.,0.,1.))


def posture_effort(pose:Pose)->dict:
    hands=[angles_for(pose,i) for i in range(2)]
    ratios=np.array([abs(h[a])/(LIMITS[a][1] if h[a]>=0 else -LIMITS[a][0])
                     for h in hands for a in ALIASES if a!="forearm.roll"])
    end_range=float(np.mean(ratios**4))
    peak=float(np.max(ratios)**4)
    wrist=float(np.mean([(max(abs(h["wrist.flex"])-15.,0.)/35.)**2
                         +.5*(h["wrist.side"]/(15. if h["wrist.side"]>=0 else 25.))**2
                         +.25*(h["forearm.roll"]/80.)**4 for h in hands]))
    coupling=float(np.mean([((h[f"{f}.dip"]-(2/3)*h[f"{f}.pip"])/45.)**2
                            for h in hands for f in ("index","middle","ring","little")]))
    spread=float(np.mean([(h[f"{f}.spread"]/spread_limit(h[f"{f}.mcp"]))**4
                          for h in hands for f in ("index","middle","ring","little")]))
    curls=[float(np.mean([np.clip((h[f"{f}.mcp"]/70.+h[f"{f}.pip"]/95.)/2.,0.,1.)
                         for f in ("index","middle","ring","little")])) for h in hands]
    tenodesis=float(np.mean([((max(h["wrist.flex"],0.)/50.)*curl
                             +(max(-h["wrist.flex"],0.)/50.)*(1.-curl))**2
                            for h,curl in zip(hands,curls)]))
    stress=.6*end_range+.25*peak+.7*wrist+.2*coupling+.2*spread+.25*tenodesis
    return {"ease":float(np.exp(-stress)),"stress":stress,"end_range":end_range,"peak_end_range":peak,
            "wrist_posture":wrist,"soft_dip_pip_coupling":coupling,"mcp_spread_loading":spread,"wrist_finger_tendon_posture":tenodesis,
            "scope":"literature-informed posture surrogate; coefficients uncalibrated; not pain, muscle force or full RULA"}


def meshes_for(rig:NativeRig,pose:Pose)->dict[str,Mesh]:
    return {side:rig.mesh(angles_for(pose,i),side,pose.positions[i],
                         Rotation.from_quat(pose.quaternions[i],scalar_first=True).as_matrix())
            for i,side in enumerate(SIDES)}


def adjacent_faces_overlap(vertices,faces,pairs)->bool:
    """Ignore only the shared topological boundary, not an entire adjacent face."""
    a,b=faces[pairs[:,0]],faces[pairs[:,1]]
    equal=a[:,:,None]==b[:,None,:]
    ma,mb=equal.any(2),equal.any(1);count=ma.sum(1)
    if (count==3).any():return True
    for shared_count in (1,2):
        selected=count==shared_count
        if not selected.any():continue
        av,bv=vertices[a[selected]],vertices[b[selected]]
        am,bm=ma[selected],mb[selected]
        origin=av[np.arange(len(av)),am.argmax(1)]
        va,vb=av-origin[:,None],bv-origin[:,None]
        na=np.cross(va[:,1]-va[:,0],va[:,2]-va[:,0])
        nb=np.cross(vb[:,1]-vb[:,0],vb[:,2]-vb[:,0])
        na/=np.linalg.norm(na,axis=1,keepdims=True);nb/=np.linalg.norm(nb,axis=1,keepdims=True)
        coplanar=(np.linalg.norm(np.cross(na,nb),axis=1)<1e-7)&(abs(np.einsum("nij,nj->ni",vb,na)).max(1)<1e-9)
        for i in np.flatnonzero(coplanar):
            drop=int(np.argmax(abs(na[i])))
            ta=np.delete(va[i],drop,axis=1).astype(np.float32)
            tb=np.delete(vb[i],drop,axis=1).astype(np.float32)
            area,_=cv2.intersectConvexConvex(ta,tb)
            if area>1e-12:return True
        if shared_count==2:continue
        # Non-coplanar faces sharing one vertex overlap only if both plane-cut
        # segments extend away from that vertex in the same direction.
        aa=va[~am].reshape(-1,2,3);bb=vb[~bm].reshape(-1,2,3)
        da=np.einsum("nij,nj->ni",aa,nb);db=np.einsum("nij,nj->ni",bb,na)
        da[abs(da)<1e-10]=0;db[abs(db)<1e-10]=0
        cut=(~coplanar)&(da.min(1)<=0)&(da.max(1)>=0)&(db.min(1)<=0)&(db.max(1)>=0)
        cut&=(abs(da[:,1]-da[:,0])>1e-12)&(abs(db[:,1]-db[:,0])>1e-12)
        if cut.any():
            da,db,aa,bb=da[cut],db[cut],aa[cut],bb[cut]
            qa=(da[:,1,None]*aa[:,0]-da[:,0,None]*aa[:,1])/(da[:,1]-da[:,0])[:,None]
            qb=(db[:,1,None]*bb[:,0]-db[:,0,None]*bb[:,1])/(db[:,1]-db[:,0])[:,None]
            if (np.sum(qa*qb,axis=1)>1e-14).any():return True
    return False


class NativeGate:
    def __init__(self,rig:NativeRig):
        self.rig=rig;self.cache:set[tuple[str,bytes]]=set()
        dominant=rig.weights.argmax(1)
        rigid=(rig.weights.max(1)>1.-1e-12)
        face_bones=dominant[rig.faces]
        self.rigid_faces=np.where(rigid[rig.faces].all(1)&(face_bones==face_bones[:,0,None]).all(1),
                                  face_bones[:,0],-1)
        self.rigid_verified:set[str]=set()

    def self_check(self,mesh:Mesh,key:tuple[str,bytes]):
        if key in self.cache:return
        triangles=mesh.vertices[mesh.faces]
        area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
        if np.min(area)<1e-12:raise InvalidPose("skin triangle degenerates at this pose")
        obj=collision_object(mesh);result=fcl.CollisionResult()
        count=fcl.collide(obj,obj,fcl.CollisionRequest(num_max_contacts=180000,enable_contact=True),result)
        if count>=180000:raise InvalidPose("skin intersection work budget exhausted")
        pairs=np.unique(np.array([(c.b1,c.b2) for c in result.contacts if c.b1<c.b2],dtype=int),axis=0)
        if len(pairs):
            a,b=mesh.faces[pairs[:,0]],mesh.faces[pairs[:,1]]
            adjacent=(a[:,:,None]==b[:,None,:]).any(axis=(1,2))
            if not adjacent.all():
                i=np.flatnonzero(~adjacent)[0]
                labels=sorted(set(mesh.labels[np.r_[a[i],b[i]]]))
                raise InvalidPose(f"{mesh.hand} skin self-intersection: {', '.join(labels)}")
            check_pairs=pairs
            if mesh.hand in self.rigid_verified:
                bone_a,bone_b=self.rigid_faces[pairs[:,0]],self.rigid_faces[pairs[:,1]]
                # A pair bound entirely to one bone has a shared rigid transform.
                # Its already-checked intersection status cannot change with pose.
                check_pairs=pairs[(bone_a<0)|(bone_a!=bone_b)]
            if len(check_pairs) and adjacent_faces_overlap(mesh.vertices,mesh.faces,check_pairs):
                raise InvalidPose(f"{mesh.hand} adjacent skin faces overlap beyond their shared boundary")
        self.rigid_verified.add(mesh.hand)
        if len(self.cache)>=192:self.cache.clear()
        self.cache.add(key)

    def check(self,pose:Pose,meshes:dict[str,Mesh]|None=None):
        Pose.parse(pose.dump())
        for i,side in enumerate(SIDES):
            values=angles_for(pose,i)
            for alias,value in values.items():
                low,high=LIMITS[alias]
                if not low-1e-8<=value<=high+1e-8:
                    raise InvalidPose(f"{side} {alias} outside conservative range [{low}, {high}] degrees")
            for digit in ("index","middle","ring","little"):
                maximum=spread_limit(values[f"{digit}.mcp"])
                if abs(values[f"{digit}.spread"])>maximum+1e-8:
                    raise InvalidPose(f"{side} {digit}: MCP spread limited to +/-{maximum:.1f} degrees at this flexion")
        if np.max(abs(pose.positions[:,:2]))>1.1 or (pose.positions[:,2]<.25).any() or (pose.positions[:,2]>2.15).any():
            raise InvalidPose("wrist workspace: x/y +/-1.1m, z .25..2.15m")
        if np.linalg.norm(pose.positions[0]-pose.positions[1])>1.1:
            raise InvalidPose("wrists must remain within 1.1m")
        meshes=meshes if meshes is not None else meshes_for(self.rig,pose)
        for i,side in enumerate(SIDES):
            mesh=meshes[side]
            if mesh.vertices[:,2].min()<.005 or mesh.vertices[:,2].max()>2.495:
                raise InvalidPose("skin intersects a screen/lamp clearance plane")
            intrinsic=pose.q.reshape(2,-1)[i].copy()
            intrinsic[INDEX["forearm.roll"]]=0.
            self.self_check(mesh,(side,np.round(intrinsic,12).tobytes()))
        a,b=meshes["right"],meshes["left"]
        if (np.minimum(a.vertices.max(0),b.vertices.max(0))>=np.maximum(a.vertices.min(0),b.vertices.min(0))).all():
            result=fcl.CollisionResult()
            if fcl.collide(collision_object(a),collision_object(b),fcl.CollisionRequest(),result):
                raise InvalidPose("hand/forearm surfaces intersect; separate them")
            if _inside(a.vertices[0],b) or _inside(b.vertices[0],a):
                raise InvalidPose("one closed arm-hand surface is inside the other")
        return meshes

    def motion(self,start:Pose,end:Pose):
        previous=self.check(start);final=self.check(end)
        rotation=2*np.arccos(np.clip(abs(np.sum(start.quaternions*end.quaternions,axis=1)),0,1))
        extent=max(float(np.degrees(abs(end.q-start.q)).max()),float(np.degrees(rotation).max()))
        translation=float(np.linalg.norm(end.positions-start.positions,axis=1).max())
        n=max(1,int(np.ceil(extent/.5)),int(np.ceil(translation/.0005)))
        if n>1024:raise InvalidPose("motion too large to check; use smaller increments")
        stack=[(i/n,(i+1)/n) for i in reversed(range(n))];samples=0;maximum=0.
        while stack:
            a,b=stack.pop();pose=interpolate(start,end,b)
            meshes=final if b==1 else meshes_for(self.rig,pose)
            displacement=max(float(np.linalg.norm(meshes[h].vertices-previous[h].vertices,axis=1).max()) for h in SIDES)
            if displacement>.0005+1e-12:
                if b-a<1e-6:raise InvalidPose("motion resolution failed; use smaller increments")
                mid=(a+b)/2;stack.extend(((mid,b),(a,mid)));continue
            self.check(pose,meshes);previous=meshes;samples+=1;maximum=max(maximum,displacement)
            if samples>1024:raise InvalidPose("motion-check budget exhausted; use smaller increments")
        return {"surface_samples":samples,"max_surface_step_mm":1000*maximum,"joint_step_limit_degrees":.5,
                "scope":"continuous native mesh; finite-resolution motion checks, not a continuum proof"}


def hand_view(meshes:dict[str,Mesh],view:str):
    az,el={"front":(0,0),"back":(180,0),"side":(90,0),"top":(0,75),"iso":(25,15)}[view]
    visible=[m for m in meshes.values() if m.vertices[m.labels!="forearm",1].max()>-.45]
    points=np.vstack([m.vertices[m.labels!="forearm"] for m in visible or list(meshes.values())])
    center=(points.min(0)+points.max(0))/2;radius=np.linalg.norm(points.max(0)-points.min(0))/2
    az,el=np.radians([az,el]);eye=center+max(.35,3.6*radius)*np.array([np.sin(az)*np.cos(el),np.sin(el),np.cos(az)*np.cos(el)])
    palette={"thumb":[215,88,65],"index":[226,158,58],"middle":[219,192,79],"ring":[96,170,111],"pinky":[88,134,193]}
    vertices=[];normals=[];faces=[];colors=[];offset=0
    for mesh in meshes.values():
        color=np.tile([222,168,128] if mesh.hand=="right" else [189,138,104],(len(mesh.vertices),1))
        for digit,tone in palette.items():color[mesh.labels==digit]=tone
        vertices.append(mesh.vertices);normals.append(mesh.normals);faces.append(mesh.faces+offset);colors.append(color)
        offset+=len(mesh.vertices)
    return _raster(np.vstack(vertices),np.vstack(normals),np.vstack(faces),np.vstack(colors),eye,center,512,384)


class NativeEnvironment:
    def __init__(self,task:Task,log_dir:Path|None=None,max_calls=100,max_images=32):
        self.task=task;self.rig=NativeRig();self.gate=NativeGate(self.rig)
        self.pose=Pose(np.zeros(46),np.array([[.1,-.12,1.3],[-.1,-.12,1.3]]),np.array([[1.,0,0,0],[1.,0,0,0]]))
        self.gate.check(self.pose);self.initial=self.pose.copy();self.history:list[Pose]=[]
        self.calls,self.images,self.max_calls,self.max_images=0,0,max_calls,max_images
        self.done=False;self.final=None;self.lock=threading.RLock();self.log_dir=log_dir
        self.hold=posture_effort(self.pose);self.steps=[];self.start_time=time.monotonic();self.version=fingerprint()
        self._metrics=None
        if log_dir:
            (log_dir/"images").mkdir(parents=True,exist_ok=True);(log_dir/"states").mkdir(exist_ok=True)
            (log_dir/"task.json").write_text(json.dumps(task.metadata,indent=2))
            wall(task.mask).save(log_dir/"target.png")
        self._record("reset",{},True,"Native Ultraleap rest pose.",None,self.pose,self.pose,None)

    def meshes(self,pose:Pose|None=None):
        return meshes_for(self.rig,self.pose if pose is None else pose)

    def metrics(self):
        if self._metrics is None:
            self._metrics=grade(shadow(self.meshes()),self.task,self.hold["ease"])
            self._metrics.update({"hand_model":MODEL_NAME,"effort_components":self.hold})
        return dict(self._metrics)

    def observe(self,view="iso"):
        if view not in ("front","back","side","top","iso"):raise InvalidPose("unknown camera view")
        meshes=self.meshes();actual=shadow(meshes)
        target=np.array(wall(self.task.mask,"FIXED TARGET"));target[~self.task.known]=[175,175,175]
        panels=[Image.fromarray(target),wall(actual,"YOUR SHADOW"),
                comparison(actual,self.task.mask,self.task.known),hand_view(meshes,view)]
        image=Image.new("RGB",(1024,768),"white")
        for i,panel in enumerate(panels):image.paste(panel,((i%2)*512,(i//2)*384))
        return image

    def describe(self):
        result={"frame_m":[-.4,.4,-.3,.3],"lamp_m":[0,0,2.5],"hand_model":MODEL_NAME,
                "joint_reference":"degrees relative to the unchanged native relaxed bind pose",
                "calls_left":self.max_calls-self.calls,"images_left":self.max_images-self.images,
                "effort":self.hold,"hands":{}}
        for i,side in enumerate(SIDES):
            values=angles_for(self.pose,i)
            wrist=self.rig.transforms(values)[self.rig.index["L_Wrist"]]
            from .ultraleap import BASIS
            basis=BASIS.copy()
            if side=="right":basis[0]*=-1
            rotation=Rotation.from_quat(self.pose.quaternions[i],scalar_first=True).as_matrix()
            directions=rotation@basis@wrist[:3,:3]
            result["hands"][side]={"anchor_m":self.pose.positions[i].round(4).tolist(),
                "palm_toward":directions[:,1].round(4).tolist(),"fingers_toward":directions[:,0].round(4).tolist(),
                "quaternion_wxyz":self.pose.quaternions[i].round(5).tolist(),
                "joints":{alias:{"deg":round(value,3),"range":list(LIMITS[alias])} for alias,value in values.items()},
                "current_spread_limits":{f:round(spread_limit(values[f"{f}.mcp"]),2) for f in ("index","middle","ring","little")}}
        return result

    def call(self,name,arguments=None)->Output:
        with self.lock:
            if self.done:return Output("Episode already ended.")
            self.calls+=1;before=self.pose.copy();image=None;certificate=None;ok=False
            args={} if arguments is None else arguments
            try:
                if name not in TOOL_SPECS:raise InvalidPose(f"unknown tool {name}")
                props=TOOL_SPECS[name][1]["properties"]
                if not isinstance(args,dict) or set(args)!=set(props):raise InvalidPose(f"{name} requires exactly {list(props)}")
                if name=="describe":text=json.dumps(self.describe())
                elif name=="look":
                    if self.images>=self.max_images:raise InvalidPose("image budget exhausted")
                    image=self.observe(args["view"]);self.images+=1
                    text="Fixed target | shadow | difference | native hand close-up. Red missing, blue extra, gray unknown."
                elif name=="submit":
                    self.done=True;self.final=self.metrics();text="Submitted."
                else:
                    candidate=self.pose.copy()
                    if name=="undo":
                        if not self.history:raise InvalidPose("nothing to undo")
                        candidate=self.history[-1].copy()
                    else:
                        side=args["hand"]
                        if side not in SIDES:raise InvalidPose("hand must be right or left")
                        i=SIDES.index(side)
                        if name=="set_fingers":
                            values=args["joints"]
                            if not isinstance(values,dict) or not values:raise InvalidPose("joints must be a nonempty object")
                            for alias,value in values.items():
                                if alias not in INDEX or type(value) not in (int,float) or not np.isfinite(value):
                                    raise InvalidPose("unknown joint or non-finite angle")
                                j=i*len(ALIASES)+INDEX[alias];angle=np.radians(value)
                                if abs(angle-self.pose.q[j])>np.radians(45)+1e-12:raise InvalidPose("maximum change 45 degrees per joint")
                                candidate.q[j]=angle
                        elif name=="move_hand":
                            delta=finite_array(args["delta"],(3,),"delta")
                            if np.linalg.norm(delta)>.15000001:raise InvalidPose("maximum translation 15cm")
                            candidate.positions[i]+=delta
                        else:
                            axis=finite_array(args["axis"],(3,),"axis");degrees=args["degrees"]
                            if type(degrees) not in (int,float) or not np.isfinite(degrees) or abs(degrees)>45 or np.linalg.norm(axis)<1e-9:
                                raise InvalidPose("nonzero axis and rotation -45..45 degrees required")
                            rotation=Rotation.from_rotvec(axis/np.linalg.norm(axis)*np.radians(degrees))
                            candidate.quaternions[i]=(rotation*Rotation.from_quat(candidate.quaternions[i],scalar_first=True)).as_quat(scalar_first=True)
                    self.gate.check(candidate);certificate=self.gate.motion(self.pose,candidate)
                    if name=="undo":self.history.pop()
                    else:self.history.append(self.pose.copy())
                    self.pose=candidate;self.hold=posture_effort(candidate);self._metrics=None
                    text=f"Accepted; {certificate['surface_samples']} checked skin states. Posture ease {self.hold['ease']:.3f} (not measured pain)."
                ok=True
            except InvalidPose as error:
                text=f"REFUSED: {error}. State unchanged.";LOG.info("%s %s",name,text)
            if self.calls>=self.max_calls and not self.done:
                self.done=True;self.final=self.metrics();text+=" Call budget exhausted; submitted."
            text+=f" [calls {self.calls}/{self.max_calls}; images {self.images}/{self.max_images}]"
            self._record(name,args,ok,text,image,before,self.pose,certificate)
            if self.done and self.log_dir:(self.log_dir/"final.json").write_text(json.dumps(self.final,indent=2))
            return Output(text,image)

    def _record(self,tool,args,ok,text,image,before,after,certificate):
        i=len(self.steps)
        record={"i":i,"tool":tool,"arguments":args,"accepted":ok,"text":text,
                "t":round(time.monotonic()-self.start_time,3),"before_hash":state_hash(before),"after_hash":state_hash(after),
                "before":before.dump(),"after":after.dump(),"certificate":certificate,"metrics":self.metrics(),
                "env_fingerprint":self.version,"hand_model":MODEL_NAME,
                "image":f"images/{i:03d}.png" if image is not None else None}
        self.steps.append(record)
        if self.log_dir:
            if image is not None:image.save(self.log_dir/record["image"])
            (self.log_dir/"states"/f"{i:03d}.json").write_text(json.dumps(after.dump()))
            with (self.log_dir/"trajectory.jsonl").open("a") as stream:stream.write(json.dumps(record,allow_nan=False)+"\n")
