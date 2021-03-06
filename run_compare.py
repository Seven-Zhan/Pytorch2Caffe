import os
import argparse
import subprocess



if __name__ == '__main__':

    parser = argparse.ArgumentParser('Convert pytorch model to caffe format')
    parser.add_argument('-t', '--task', default='task', type=str, help='type of task: fic or feh')
    
    args, unparsed = parser.parse_known_args()

    script_path = os.path.join('result', args.task, 'compare.py')
    subprocess.run(['python', script_path])
