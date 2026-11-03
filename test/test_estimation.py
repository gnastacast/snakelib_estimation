import os
import subprocess
import tempfile

from ament_index_python.packages import get_package_share_directory
import numpy as np
import pytest  # noqa: F401
from urdf_parser_py.urdf import URDF
import xacro

from snakelib_estimation import Kinematics

def spoof_xacro(snake_type, module_names, kinematic_only=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.abspath(os.path.join(
            tmpdir,
            'snake.xacro'
        ))

        with open(yaml_path, 'w') as f:
            f.write(f'snake_type: {snake_type}\n')
            f.write('modules: [\n')
            for name in module_names:
                f.write('{ name: %s },\n' % (name))
            f.seek(f.tell() - 2)
            f.write('\n]\n')

        urdf_path = os.path.join(
            get_package_share_directory('snakelib_description'),
            f'{snake_type}_snake',
            f'{snake_type}_snake.xacro'
        )
        return xacro.process_file(
            urdf_path,
            mappings={
                'yaml_file': yaml_path,
                'kinematic_only': 'true' if kinematic_only else 'false',
            }
        ).toxml()

def test_kinematics():
    # Check xacro generation
    xml = spoof_xacro('REU', ['RUHA-001', 'RUA-001', 'RUB-002', 'RUC-003', 'RUTA-001'])
    kin = Kinematics(xml)
    kin.update([0,0,0])
    frames = kin.fk()
    print(frames)
    # assert np.allclose(frames[0], np.eye(4))
    dists = np.linalg.norm(frames[1:][0:3,3] - frames[:-2][0:3,3], axis=1)
