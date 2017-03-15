import scipy.ndimage
import numpy as np

def _create_elastic_distortions(sigma, alpha):
    # Initializing with uniform distribution between -1 and 1
    x = np.random.uniform(-1, 1, size=(28, 28))
    y = np.random.uniform(-1, 1, size=(28, 28))

    # Convolving with a Gaussian filter
    x = scipy.ndimage.filters.gaussian_filter(x, sigma)
    y = scipy.ndimage.filters.gaussian_filter(y, sigma)

    # Multiplying elementwise with alpha
    x = np.multiply(x, alpha)
    y = np.multiply(y, alpha)

    return x, y
    
def _create_elastic_filters(n_filters, sigma=4.0, alpha=8.0):
    return [_create_elastic_distortions(sigma, alpha) for _ in xrange(n_filters)]
    
# Applies an elastic distortions filter to image
def _apply_elastic_distortions(image, filter):
    # Ensures images are of matrix representation shape
    image = np.reshape(image, (28, 28))
    res = np.zeros((28, 28))

    # filter will come out of _create_elastic_filter
    f_x, f_y = filter

    for i in xrange(28):
        for j in xrange(28):
            dx = f_x[i][j]
            dy = f_y[i][j]

            # These two variables help refactor the code
            # They are a little mind tricky; don't hesitate to take a pen and paper to visualize them
            x_offset = 1 if dx >= 0 else -1
            y_offset = 1 if dy >= 0 else -1

            # Retrieving the two closest x and y of the pixels near where the arrow ends
            y1 = j + int(dx) if 0 <= j + int(dx) < 28 else 0
            y2 = j + int(dx) + x_offset if 0 <= j + int(dx) + x_offset < 28 else 0
            x1 = i + int(dy) if 0 <= i + int(dy) < 28 else 0
            x2 = i + int(dy) + y_offset if 0 <= i + int(dy) + y_offset < 28 else 0

            # Horizontal interpolation : for both lines compute horizontal interpolation
            _interp1 = min(max(image[x1][y1] + (x_offset * (dx - int(dx))) * (image[x2][y1] - image[x1][y1]), 0), 1)
            _interp2 = min(max(image[x1][y2] + (y_offset * (dx - int(dx))) * (image[x2][y2] - image[x1][y2]), 0), 1)

            # Vertical interpolation : for both horizontal interpolations compute vertical interpolation
            interpolation = min(max(_interp1 + (dy - int(dy)) * (_interp2 - _interp1), 0), 1)

            res[i][j] = interpolation

    return res
    
# Creates and apply elastic distortions to the input
# images: set of images; labels: associated labels
def expand_dataset(images, labels, n_distortions):
    distortions = _create_elastic_filters(n_distortions)
    
    new_images_batch = np.array(
        [_apply_elastic_distortions(image, filter) for image in images for filter in distortions])
    new_labels_batch = np.array([label for label in labels for _ in distortions])

    # We don't forget to return the original images and labels (hence concatenate)
    return np.concatenate([np.reshape(images, (-1, 28, 28)), new_images_batch]), \
           np.concatenate([labels, new_labels_batch])


# Returns a batch from (images, labels) begining at begin ending at begin+batch_size-1 (included)
def get_batch(images, labels, begin, batch_size):
    return images[begin: begin+batch_size], labels[begin: begin+batch_size]


def main():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('tmp/data', one_hot=True)

    print 'lala'
    print len(expand_dataset(mnist.train.images, mnist.train.labels, 3)[0])

if __name__ == '__main__':
    main()