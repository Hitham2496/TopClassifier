"""
=============================================
Machine learning with image pre-processing
=============================================

Supervised machine learning method with image
preprocessing for wide jets to tag those
originating from top quark decays by
examining their pT-distribution in eta-phi
_____________________________________________
"""
# Import modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model
from itertools import groupby

SIZE = 72 
INTERVAL = 1.5

def unpack(filename):
    """
    Unpacks a file containing (eta, phi, pT) triples into
    one numpy array per jet with tagging info at the end
    """
    array_list = []
    with open(filename) as f:
        # create a new array every new line
        for k, g in groupby(f, lambda x: x.startswith('\n')):
            if not k:
                # combine each array of triples to a list of arrays
                array_list.append(np.array([[float(x) for x in d.split()]
                                           for d in g if len(d.strip())]))
    return array_list

def crop(image,c_x,c_y):
    """
    Crops an image about the central element
    """
    y,x = image.shape
    startx = x//2-(c_x//2)
    starty = y//2-(c_y//2)
    return image[starty:starty+c_y,startx:startx+c_x]

def generate_image(coords):
    """
    Generates an un-normalised eta-phi image with preprocessing
    stages applied as follows:
    1) the hardest constituent is placed at the centre (0,0)
    2) the image is rotated such that the second hardest
       constituent has phi = 0
    3) flip the image if the third hardest constituent has
       phi < 0
    """
    image=np.zeros((SIZE, SIZE))
    N = np.linspace(-INTERVAL, INTERVAL, SIZE)

    # locate first, second and third hardest constituents
    max1 = np.amax(coords[2:coords.shape[0]:3])
    max1_index = np.where(coords == max1)
    eta_max1 = coords[max1_index[0]-1]
    phi_max1 = coords[max1_index[0]-2]


    coords_copy = coords
    coords_copy[max1_index] = 0.
    max2 = np.amax(coords_copy[2:coords.shape[0]:3])
    max2_index = np.where(coords_copy == max2)[0]
    coords_copy[max2_index] = 0.
    max3 = np.amax(coords_copy[2:coords.shape[0]:3])

    # first process: central element is the maximum
    image[int(SIZE/2), int(SIZE/2)] = max1
    if coords[max2_index[0]-1] != 0.:

        # second process: rotate such that max2 is at phi = 0
        eta_old = coords[max2_index[0]-1]
        phi_old = coords[max2_index[0]-2]

        angle = np.pi/2. - np.arctan2(eta_old, phi_old)

        for i in range(0, coords.shape[0]-1, 3):
            # translate by eta, phi of max1
            coords[i+1] -= eta_max1
            coords[i] -= phi_max1
            eta_tmp = coords[i+1]
            phi_tmp = coords[i]

            # rotate to give max2 phi = 0
            coords[i+1] = -np.sin(angle)*eta_tmp + np.cos(angle)*phi_tmp
            coords[i] = np.sin(angle)*phi_tmp + np.cos(angle)*eta_tmp

    for i in range(0, coords.shape[0]-1, 3):
        # fit (eta,phi) to the grid: eta = [i], phi = [i+1], pT = [i+2]
        if (coords[i+2] != max1 and not
            (abs(coords[i]) > INTERVAL or abs(coords[i+1]) > INTERVAL)):
            index_y = np.abs(N - coords[i]).argmin()
            index_x = np.abs(N - coords[i+1]).argmin()
            image[index_y, index_x] += coords[i+2]

    # third process: flip image such that max3 has phi > 0
    max3_index = np.where(image == max3)
    if max3_index[0].any() < SIZE/2:
        image = np.fliplr(image)

    return image

def normalise(image):
    """
    Normalises any array in the l-1 norm as
    if it were a column vector
    """
    pT_tot = np.sum(image)
    image = image/pT_tot
    return image

def generate_image_array(filename):
    """
    Generates normalised images from a data
    file containing triples and tagging info
    """
    f = unpack(filename)
    img = np.zeros((len(f), SIZE, SIZE))
    tags = np.zeros((len(f), 1))
    for i in range(0, img.shape[0]):
    # generate cropped image (in case boundaries above are changed)
        img[i] = normalise(crop(generate_image((f[i][0])), SIZE, SIZE))
        tags[i] = np.asarray(f[i])[:,-1]
    return img, tags

def average_images(images, tags):
    """
    Averages the images of top and non-top candidates
    """
    top_average = np.zeros((SIZE, SIZE))
    nontop_average = np.zeros((SIZE, SIZE))
    top_count = 0
    nontop_count = 0
    for i in range(0, images.shape[0]):
        if tags[i] == 0.:
            nontop_average += images[i]
            nontop_count += 1.
        elif tags[i] == 1.:
            top_average += images[i]
            top_count += 1.
    top_average = top_average/top_count
    nontop_average = nontop_average/nontop_count
    return nontop_average, top_average

def false_pos(y_true,y_pred):
    """
    Gives the number of false positive results
    """
    res = 0.
    for i in range(y_pred.shape[0]):
        if y_pred[i]==1. and y_true[i]!=y_pred[i]:
            res += 1.
    return res

def logit_tt(img, y):
    """
    Uses logistic regression to tag top jet candidates
    """
    n, nx, ny = img.shape
    img = img.reshape((n, nx*ny))
    y = y.reshape((n))
    # split into testing and training datasets
    img_train, img_test, y_train, y_test = model_selection.train_test_split(
            img, y, test_size = 0.5, shuffle = True)

    # classify by logistic regression
    clf = linear_model.LogisticRegression()
    clf.fit(img_train, y_train)
    y = clf.predict(img_test)
    clf_s = clf.score(img_test,y_test)
    fp = false_pos(y_test,y)/float(int(n/2))
    return clf, y, clf_s, fp


def main():
    images, y = generate_image_array('image_bin2.dat')
    x_n, x_t = average_images(images, y)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.imshow(x_t, interpolation ="gaussian", aspect = 'equal', cmap = 'magma',
            vmin = 1e-7, vmax = 1e-3, extent=[-INTERVAL,INTERVAL,-INTERVAL,INTERVAL])
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\eta$')
    plt.savefig('top_jets.pdf', bbox_inches = 'tight')
    plt.show()

    plt.imshow(x_n, interpolation ="gaussian", aspect = 'equal', cmap = 'magma',
               vmin = 1e-7, vmax = 2e-3, extent=[-INTERVAL,INTERVAL,-INTERVAL,INTERVAL])
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\eta$')
    plt.savefig('non_top_jets.pdf', bbox_inches = 'tight')
    plt.show()

    clf, y, clf_s, fp = logit_tt(images, y)
    print("Classification score = %s" % round(clf_s, 6) )
    print("False-positive identification rate = %s" % round(fp, 6) )


if __name__ == "__main__":
    print(__doc__)
    main()

