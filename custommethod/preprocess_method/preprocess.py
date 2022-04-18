'''
Requires the LASTools suite (http://lastools.org) installed and the .exe is inside PATH!
'''
from glob import glob
from numpy.random import randint, random, uniform
from os import remove
from subprocess import check_output, DEVNULL

def tileScene(scene = 'scene1.las'):
    log = check_output(f'lastile -cpu64 -i {scene} -tile_size 10 -buffer 5 -reversible')

def randomAugment():
    for tile in glob('*.las'):
        for num in range(5):
            transforms = ''
            # 50% chance to add each transform

            if random() >= 0.5:
                transforms += f' -scale_z {uniform(.8, 1.2)}'
            if random() >= 0.5:
                transforms += f' -rotate_xy {uniform(-18, 18)} 0 0'
            if random() >= 0.5:
                transforms += f' -rotate_yz {uniform(-12, 12)} 0 0'
            if random() >= 0.5:
                transforms += f' -rotate_xz {uniform(-12, 12)} 0 0'
            if random() >= 0.5:
                transforms += f' -translate_then_scale_y {uniform(-.1, .1)} {uniform(.8, 1.2)}'
            if random() >= 0.5:
                transforms += f' -transform_affine {uniform(-9, 9)},{uniform(-9, 9)},{uniform(0, 1000)},{uniform(0, 1000)}'
            # log = check_output(f'las2las -cpu64 -scale_z 0.3048 -rotate_xy 15.0 620000 4100000 -translate_then_scale_y -0.5 1.001  -transform_affine 0.9999652,0.903571,171.67,736.26')
            log2 = check_output(f'las2las -cpu64 -i {tile} -o {tile.replace(".las","")}_augment{num:02}.las {transforms}',stderr=DEVNULL)

def voxelise():
    for tile in glob('*.las'):
        try:
        	log2 = check_output(f'las2las  -cpu64 -keep_class 2  -i {tile.replace(".las","")}.las -o  {tile.replace(".las","")}a.las',stderr=DEVNULL)
        except:
        	pass
        try:
        	log2 = check_output(f'las2las  -cpu64 -keep_class 3  -i {tile.replace(".las","")}.las -o  {tile.replace(".las","")}b.las',stderr=DEVNULL)
        except:
        	pass
        try:
        	log2 = check_output(f'las2las  -cpu64 -keep_class 9  -i {tile.replace(".las","")}.las -o  {tile.replace(".las","")}c.las',stderr=DEVNULL)
        except:
        	pass

        try:
        	log2 = check_output(f'lasvoxel  -cpu64 -i {tile.replace(".las","")}a.las -o  {tile.replace(".las","")}aa.las -step 0.1',stderr=DEVNULL)
        except:
        	pass
        try:
        	log2 = check_output(f'lasvoxel  -cpu64 -i {tile.replace(".las","")}b.las -o  {tile.replace(".las","")}bb.las  -step 0.1',stderr=DEVNULL)
        except:
        	pass
        try:
        	log2 = check_output(f'lasvoxel  -cpu64 -i {tile.replace(".las","")}c.las -o  {tile.replace(".las","")}cc.las -step 0.1',stderr=DEVNULL)
        except:
        	pass

        try:
        	log2 = check_output(f'las2las -set_classification 2 -i {tile.replace(".las","")}aa.las -o  {tile.replace(".las","")}aaa.las',stderr=DEVNULL)
        except:
        	pass
        try:
        	log2 = check_output(f'las2las -set_classification 3 -i {tile.replace(".las","")}bb.las -o  {tile.replace(".las","")}bbb.las',stderr=DEVNULL)
        except:
        	pass
        try:
        	log2 = check_output(f'las2las -set_classification 9 -i {tile.replace(".las","")}cc.las -o  {tile.replace(".las","")}ccc.las',stderr=DEVNULL)
        except:
        	pass
        
        try:
        	log2 = check_output(f'lasmerge -i {tile.replace(".las","")}aaa.las {tile.replace(".las","")}bbb.las {tile.replace(".las","")}ccc.las -o {tile.replace(".las","")}_voxelised_0.1.las',stderr=DEVNULL)
        except:
        	pass

        # Remove intermediate files
        try:
        	remove(f'{tile.replace(".las","")}a.las')
        except:
        	pass
        try:
        	remove(f'{tile.replace(".las","")}b.las')
        except:
        	pass
        try:
        	remove(f'{tile.replace(".las","")}c.las')
        except:
        	pass
        try:
        	remove(f'{tile.replace(".las","")}aa.las')
        except:
        	pass
        try:
        	remove(f'{tile.replace(".las","")}bb.las')
        except:
        	pass
        try:
        	remove(f'{tile.replace(".las","")}cc.las')
        except:
        	pass
        try:
        	remove(f'{tile.replace(".las","")}aaa.las')
        except:
        	pass
        try:
        	remove(f'{tile.replace(".las","")}bbb.las')
        except:
        	pass
        try:
        	remove(f'{tile.replace(".las","")}ccc.las')
        except:
        	pass
if __name__ == '__main__':
    randomAugment()