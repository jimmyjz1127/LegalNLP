import sys
import json

    
def preview(path):
    with open(path, 'r') as file:
        count = 0
        for line in file:
            json_line = json.loads(line)

            print(json_line.keys())

            # for key,val in json_line.items():
            #     print(f'[{key}]')
            #     print(val)
            #     print('=======================================================================')

            return

if __name__ == '__main__':
    preview(sys.argv[1])