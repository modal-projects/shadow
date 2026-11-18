# Licenses

ShadowBench combines code, data made by Perpetual Research, public-domain book plates and a third-party hand model.

| Part | License |
| --- | --- |
| Code: `shadowbench/`, `scripts/` | MIT (below) |
| Episodes (`data/episodes`, `data/steps`, `data/transcripts`), the leaderboard, the task table and the target drawings traced and certified from the plates (`data/bench/targets`, `data/bench/books/*/target.*`) | CC BY 4.0 |
| Book plates (`data/bench/books/*/source.png` and the book pages in the episodes): Henry Bursill, *Hand Shadows to be Thrown upon the Wall* (1859); William J. Hilliar, *Novel Hand Shadows* (c. 1900); Félicien Trewey, *The Art of Shadowgraphy* (c. 1900) | Public domain. Scans from [Project Gutenberg](https://www.gutenberg.org/ebooks/12962) and the [Internet Archive](https://archive.org/details/novelhandshadows00hill) ([Trewey](https://archive.org/details/artofshadowgraph00trewuoft)). |
| Hand and forearm model (`assets/ultraleap/native_rig.npz`), converted from `GenericHand_Arm.fbx` in the [Ultraleap Unity Plugin](https://github.com/ultraleap/UnityPlugin) (commit `08ea531`): units converted to metres in a wrist-local frame, skin weights normalized. The hands drawn in the episodes' images are renders of it. | Apache License 2.0, text in [`assets/ultraleap/LICENSE.md`](assets/ultraleap/LICENSE.md). |

The model messages in `data/transcripts` and the plans and closing messages in `data/episodes` were written by the
listed third-party models, and their providers' terms may apply.

## MIT License

Copyright (c) 2026 Perpetual Research

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
