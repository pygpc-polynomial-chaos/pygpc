import subprocess
import os
import sysconfig
import numpy as np


def build_pygpc_extensions_cuda():
    ext_prefix = sysconfig.get_config_var('EXT_SUFFIX').split('.')
    ext_prefix = ext_prefix[1]
    configure_command = ['cmake', '-B'+os.path.join('build', 'pygpc_extensions_cuda'),
                         '-H'+os.path.join('pckg','pygpc_extensions_cuda'),
                         '-DPROJECT_ROOT_PATH='+os.getcwd(),
                         '-DNumPy_INCLUDE_DIRS='+np.get_include(),
                         '-DEXT_PREFIX='+ext_prefix]
    subprocess.run(configure_command)
    build_commad = ['cmake', '--build', os.path.join('build', 'pygpc_extensions_cuda')]
    print(build_commad)
    subprocess.run(build_commad)


if __name__ == '__main__':
    build_pygpc_extensions_cuda()
