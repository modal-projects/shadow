"""Native Ultraleap hand/forearm deformation for the separately versioned surface trial.

No MyoHand retargeting or proxy forearm is used. The source's open elbow end is
closed locally; every other vertex, bone offset and skin weight comes from its rig.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .mesh import Mesh

ASSET=Path(__file__).resolve().parents[1]/"assets"/"ultraleap"/"native_rig.npz"
if not ASSET.is_file():ASSET=Path(__file__).resolve().parent/"data"/"ultraleap"/"native_rig.npz"
DIGITS=("thumb","index","middle","ring","pinky")
BASIS=np.array([[0.,0.,-1.],[1.,0.,0.],[0.,-1.,0.]])


class NativeRig:
    def __init__(self):
        with np.load(ASSET) as data:
            self.vertices=data["vertices"].copy()
            self.faces=data["faces"].copy()
            self.weights=data["weights"].copy()
            self.rest=data["rest"].copy()
            self.parents=data["parents"].copy()
            self.names=data["names"].tolist()
        self.index={name:i for i,name in enumerate(self.names)}
        self.inverse=np.linalg.inv(self.rest)
        self.local=self.rest.copy()
        for i,parent in enumerate(self.parents):
            if parent>=0:self.local[i]=self.inverse[parent]@self.rest[i]
        self._close_elbow()
        self.labels=np.array([next((d for d in DIGITS if d in name),"forearm" if name=="Elbow" else "palm")
                              for name in self.names])[self.weights.argmax(1)]
        self.order=[]
        pending=set(range(len(self.names)))
        while pending:
            ready=sorted(i for i in pending if self.parents[i]<0 or self.parents[i] in self.order)
            if not ready:raise ValueError("native skeleton has a cycle")
            self.order.extend(ready);pending.difference_update(ready)

    def _close_elbow(self):
        edges=np.concatenate([self.faces[:,[0,1]],self.faces[:,[1,2]],self.faces[:,[2,0]]])
        _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
        boundary=edges[counts[inverse]==1]
        if len(boundary)==0:return
        ids=np.unique(boundary);elbow=self.rest[self.index["Elbow"],:3,3]
        if abs(self.vertices[ids,0]-elbow[0]).max()>.005:
            raise ValueError("source boundary is not the expected elbow cut; cannot safely cap it")
        center=len(self.vertices)
        self.vertices=np.vstack([self.vertices,self.vertices[ids].mean(0)])
        extra=np.zeros((1,len(self.names)));extra[0,self.index["Elbow"]]=1
        self.weights=np.vstack([self.weights,extra])
        self.faces=np.vstack([self.faces,np.c_[boundary[:,1],boundary[:,0],np.full(len(boundary),center)]])
        mesh=trimesh.Trimesh(self.vertices,self.faces,process=False)
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError("native arm-hand skin is not a closed, consistently oriented surface")

    def transforms(self,angles:dict[str,float]):
        deltas=np.zeros((len(self.names),3))
        for alias,value in angles.items():
            finger,joint=alias.split(".")
            digit="pinky" if finger=="little" else finger
            if finger=="wrist":
                bone="L_Wrist";axis={"flex":2,"side":1}[joint]
            elif finger=="forearm":
                if joint!="roll":raise ValueError("unknown forearm degree of freedom")
                bone="Elbow";axis=0
            elif finger=="thumb":
                bone={"base":"L_thumb_Proximal","opposition":"L_thumb_Proximal",
                      "mcp":"L_thumb_a","ip":"L_thumb_b"}[joint]
                axis=1 if joint=="opposition" else 2
            else:
                if digit not in DIGITS:raise ValueError("unknown digit")
                bone=f"L_{digit}_"+{"mcp":"Proximal","spread":"Proximal","pip":"b","dip":"c"}[joint]
                axis=1 if joint=="spread" else 2
            deltas[self.index[bone],axis]=np.radians(value)
        if not np.isfinite(deltas).all():raise ValueError("joint angles must be finite")
        local=self.local.copy()
        local[:,:3,:3]=local[:,:3,:3]@Rotation.from_euler("XYZ",deltas).as_matrix()
        posed=np.empty_like(local)
        for i in self.order:
            p=self.parents[i]
            posed[i]=local[i] if p<0 else posed[p]@local[i]
        return posed

    def mesh(self,angles:dict[str,float],hand="left",position=(0.,-.12,1.3),rotation=None):
        transforms=self.transforms(angles)@self.inverse
        blend=np.einsum("vj,jab->vab",self.weights,transforms)
        vertices=np.einsum("vab,vb->va",blend[:,:3,:3],self.vertices)+blend[:,:3,3]
        basis=BASIS.copy()
        if hand=="right":basis[0]*=-1
        elif hand!="left":raise ValueError("hand must be left or right")
        vertices=vertices@basis.T
        if rotation is not None:vertices=vertices@rotation.T
        vertices+=np.asarray(position)
        faces=self.faces[:,::-1] if hand=="right" else self.faces
        return Mesh(vertices,faces,hand,self.weights,self.labels)
