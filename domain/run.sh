#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LD_LIBRARY_PATH="$DIR/libs:$LD_LIBRARY_PATH"
"/DCCV/domain/build/release/domain/bin/domain" -w=9 -h=6
# -v="/DCCV/domain/video/video.avi" -id=3