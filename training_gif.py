import glob
import imageio
import os

def save_gif(folder, delimiter):
    """
        expects images to be saved like this:

        <folder> 
            <delimiter>0.png
            <delimiter>1.png
            <delimiter>2.png
    """
    def get_key(fname):
        basename = os.path.basename(fname)
        start = basename.rfind(delimiter)
        key = str(int(basename[start+len(delimiter):-len('.png')])).zfill(3)
        return key

    fnames = sorted(glob.glob('{}/{}*.png'.format(folder, delimiter)), key=get_key)
    print(fnames)
    
    if len(fnames) > 0:
        images = [imageio.imread(fname) for fname in fnames]
        imageio.mimsave(os.path.join(folder, '{}.gif'.format(delimiter)), images, fps=2)
    else:
        print('Could not find any images in {}'.format(folder))

def main():
    save_gif('vae_samples','mnist_vae_')

if __name__=="__main__":        
    main()