# TVD-SP Coverage Control Paper Code

This respository contains UCSD coverage control algorithm implementations for the TVD-SP Coverage Control paper.
3D simulations are in pycoverage_3d, while Robotarium implementations are in pycoverage_robotarium.

To install the project run
```
pip install path/to/tvd_coverage/
```
See the [Robotarium python simulator](https://github.com/robotarium/robotarium_python_simulator) and the Robotarium [website](robotarium.gatech.edu) if you want to try it for yourself. Follow their install instructions to use the pycoverage_robotarium code.

The entry point of pycoverage2d is `simulations.py` while the entry point of pycoverage_robotarium is `main_phi4.py`.

This project, as a whole, is distributed under the terms of the **GNU General Public License v3.0 or later** (GPL-3.0-or-later).

Because this project incorporates a component licensed under the GPL-3.0, the entirety of the program must be offered under the GPL-3.0, as per its copyleft terms.

However, where explicitly noted in the source file headers, individual files that do not directly import or link to the GPL-licensed code are also available under the more permissive **MIT License**. This allows users to reuse those specific files under the MIT license, provided the usage does not result in the creation of a combined work with the GPL-licensed component.

*   The full text of the GPL-3.0 is provided in the **LICENSE** file.
*   The full text of the MIT License is provided in the **LICENSE-MIT** file.
*   The full text of the license for pyvoro and voro++ is provided in the **LICENSE-THIRDPARTY** file

**Source File Licensing Breakdown:**
*   Files using or derived from the GPL-3.0 code are licensed under `GPL-3.0-or-later`.
*   Original source files are licensed under `MIT`.
*   The license for each file is explicitly stated in its header using the `# SPDX-License-Identifier` comment.