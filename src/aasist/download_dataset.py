"""
AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os

if __name__ == "__main__":
    # cmd = "curl -o ./LA.zip -# https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip\?sequence\=3\&isAllowed\=y"
    cmd = "curl -o ./LA.zip -# http://10.5.110.131:8080/LA.zip
    os.system(cmd)
    cmd = "unzip LA.zip"
    os.system(cmd)