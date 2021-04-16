# Reproducing a Deep Learning Paper
### By Asror Wali & Erwin Russel

![Header image](https://i.ibb.co/Cwhjm9T/zwaan-crf2.png)

This is a short blog about a reproduction of the paper "W-Net: A Deep Model for Fully Unsupervised Image Segmentation".
This reproduction is part of the CS4240 Deep Learning course at the TU Delft.

We were asked to choose from a list of papers, where this was the one we found most interesting.
There was a Github repository attached to the paper which we could use, in this blog we will describe the process of using this code repository and adapting it to achieve the results that were displayed in the paper.
Next to reproducing the cool segmentation abilities of the WNet architecture, we also want to see how our reproduction benchmarked against the model in the paper. There is a table explaining various benchmarks that we will go into in this blog. 

The dataset is discussed following the losses and postprocessing. We are showing preliminary results on a small dataset, since the original paper trained the model 50000 epochs on 12000 images  this was not attainable given the resources given (google cloud and educational credit). We therefore show one of the results training on a miniscule dataset size of only 10 images running with 100 epochs. 

## Dataset

W-Net uses two different datasets, they train the network using the PASCAL VOC2012 dataset and evaluate the network using the Berkeley Segmentation Database (BSDS 300 and BSDS 500).  The PASCAL VOC2012 dataset is a large visual object classes challenge which contains 11,530 images and 6,929 segmentations. BSDS300 and BSDS500 have 300 and 500 images including human-annotated segmentationas ground truth. 

## The Model

![W-Net architecture](https://raw.githubusercontent.com/AsWali/WNet/master/media/Architecture.PNG)

The W-Net model consists of 2 U-Nets, the first U-Net works as an encoder that generates a segmentated output for an input image X. The second U-Net uses this segmentated output to reconstruct the original input image X. 

To optimize these 2 U-Nets, the paper introduces 2 loss functions. First, A Soft Normalized Cut Loss(soft_n_cut_loss), to optimize the encoder and Secondly a Reconstruction Loss(rec_loss), to optimize both the encoder and the decoder. 

The `soft_n_cut_loss` simultaneously minimizes the total normalized disassociation between the groups and maximize the total normalized association within the groups. In other words, the similarity between pixels inside of the same group/segment gets maximized while the similarity between different groups/segments get minimized. 

The `rec_loss` forces the encoder to generate segmentations that contain as much information of the original input as possible. The decoder prefers a good segmentation, so the encoder is forced to meet him half way. To show this we included a image:

![A meme showing the decoder needs a good segmentation to create a reconstruction](https://raw.githubusercontent.com/AsWali/WNet/master/media/enc_dec_meme.png)


The W-Net code is publicly available on GitHub[^x1]. Although it is provided it is in an incomplete state, the 2 U-Nets have been implemented but it's missing the Relu activation and the dropout mentioned in the W-Net paper. The script provided to train the model is also not implemented. And both loss functions used in the paper are not implemented. Instead they used kernels for vertical and horizontal edge detection to optimize the encoder and an unusual mean function for the decoder.

So to reproduce the W-Net paper, we need to add all these missings elements. Most of these are easy to add, so we mostly will focus on the hard part the loss functions and the benchmarking process.

## Losses

The algorithm to train the model is described as this in the paper:
![Algorithm the use to train the model](https://raw.githubusercontent.com/AsWali/WNet/master/media/algo_1.png)

The delivered function looks like this:
```python
def train_op(model, optimizer, input, psi=0.5):
    enc = model(input, returns='enc')
    n_cut_loss=gradient_regularization(enc)*psi
    n_cut_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    dec = model(input, returns='dec')
    rec_loss=torch.mean(torch.pow(torch.pow(input, 2) + torch.pow(dec, 2), 0.5))*(1-psi)
    rec_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return model
```

So we need to implemented the losses used in the paper, the reconstruction loss (`rec_loss`) is an easy one to implement. It looks like this:

![The reconstruction loss used](https://raw.githubusercontent.com/AsWali/WNet/master/media/rec_loss.png)

So we need to minimize the distance between the original image X and the output of the decoder, given the segmentated output of the encoder.

```python
def reconstruction_loss(x, x_prime):
    criterionIdt = torch.nn.MSELoss()
    rec_loss = criterionIdt(x_prime, x)
    return rec_loss

dec = model(input, returns='dec')
rec_loss=reconstruction_loss(input, dec)
```

This works because the model implements the decoder like this: `dec=self.UDec(F.softmax(enc, 1))`. 

Add this loss to the train_op function
```python
def train_op(model, optimizer, input, psi=0.5):
    enc = model(input, returns='enc')
    n_cut_loss=gradient_regularization(enc)*psi
    n_cut_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    dec = model(input, returns='dec')
    rec_loss=reconstruction_loss(input, dec)
    rec_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return model
```

The Soft Normalized Cut Loss(`soft_n_cut_loss`) is a bit harder to implement. The function looks like this:

![The soft normalized n cut loss](https://raw.githubusercontent.com/AsWali/WNet/master/media/soft_n_cut_loss.png)

With `w(u,v)` being a weight between `(u,v)` and `p` being the probability value the encoder gives to an pixel belonging to group k, implementing `w `looks like this:

![Weight calculation](https://raw.githubusercontent.com/AsWali/WNet/master/media/weight_calc.png)


First thing we did was look at already existing implementations, of which we found two:
1. https://github.com/gr-b/W-Net-Pytorch[^x2], uses a matrix solution to compare all pixels with each other. Does not use the `radius` mentioned in the paper, although it is added as a argument. Some of the methods are also ported from a Tensorflow implementation, so the code has custom implementations for calls which already exist in the PyTorch API, like custom outer functions.
2. https://github.com/fkodom/wnet-unsupervised-image-segmentation[^x3], which uses conv2d kernels to compare the pixels, which is much more efficient when working with large images. This implementation deviates from the paper by doing an pixel average, they mention this: "Small modifications have been made here for efficiency -- specifically, we compute the pixel-wise weights relative to the class-wide average, rather than for every individual pixel."

**Instead of using these implementations, we decided to implement this loss ourselves and stay true to the paper.**

### Aproach 1: creating a weight matrix
Because we implemented this on our relatively low spec laptops we worked with `64x64` images while developing. So the matrix approach worked without giving memory issues, and was the first approach we did. The idea was similar to the first Github linked above, but the code is created on our own. The main idea was to create a weight matrix which compares all `NxN` pixels with each other. So this results in a weight matrix of size `N*NxN*N`, quite a big matrix but doable for image sizes of `64x64`. To illustrate our approach we are going to use a 3x3 matrix, lets say we have this matrix.

```python
tensor([[1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]])
```

We will flatten this image and expand it NxN times, creating this `N*NxN*N` matrix.
```python
tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9.],
        [1., 2., 3., 4., 5., 6., 7., 8., 9.],
        [1., 2., 3., 4., 5., 6., 7., 8., 9.],
        [1., 2., 3., 4., 5., 6., 7., 8., 9.],
        [1., 2., 3., 4., 5., 6., 7., 8., 9.],
        [1., 2., 3., 4., 5., 6., 7., 8., 9.],
        [1., 2., 3., 4., 5., 6., 7., 8., 9.],
        [1., 2., 3., 4., 5., 6., 7., 8., 9.],
        [1., 2., 3., 4., 5., 6., 7., 8., 9.]])
```

Now if we subtract from this matrix its own transponse we will get the difference between 2 values at position (i,j):
```python
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.],
        [-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [-2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.],
        [-3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.],
        [-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.],
        [-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.],
        [-6., -5., -4., -3., -2., -1.,  0.,  1.,  2.],
        [-7., -6., -5., -4., -3., -2., -1.,  0.,  1.],
        [-8., -7., -6., -5., -4., -3., -2., -1.,  0.]])
```
If we divide all these values by the given deviation parameter and put it on an expontial we will get the first part of the W matrix. The second part uses the radius, so that it only looks at pixels inside of the radius length.

To implement this radius constraint, we did the same thing again with the flattening and the expanding as above, but this time we created 2 matrices and used index locations as values. For example the matrix containing the X locations.
```python
tensor([[0., 1., 2.],
        [0., 1., 2.],
        [0., 1., 2.]])
```
So after flattening, expanding and substracting it own transpose we have 2 matrices. One containing the distance on the X dimension and one for the Y dimension. If we add these distance we got the manhatten distance, but we want the euclidian one

```python
Xij = (X_expand - X_expandT)
Yij = (Y_expand - Y_expandT)

sq_distance_matrix = torch.hypot(Xij, Yij)
```
So now we got the distances between pixels, we can set all distance to 0 if they are larger than the given radius using a mask. We first apply the mask to the sq_distance_matrix, divide this matrix and apply the exponential. After applying these transformations, the two matrices are multiplied to create the final weight matrix

```python
mask = sq_distance_matrix.le(radius)

C = torch.exp(torch.div(-1*(sq_distance_matrix **2), ox**2))
W_X = torch.mul(mask, C)

weights = torch.mul(W_F, W_X)
```

Now we can use this weight matrix and the results of the encoder to calculate the `soft_n_cut_loss`, but the paper mentioned that they trained on images of size `224x224`. Using the weight matrix method on images of this size result in multiple matrices of size `50176x50176`.

### Aproach 2: custom sliding window
The second idea is based on the local computation involving the radius. The weights are 0 for pixels that are farther away than the radius. So the calculations needed could be contained to a window of size `radius*2 + 1`, the plus 1 so that the kernel has an uneven size and has a good center.

Lets consider the following image:
```python
tensor([[0.3337, 0.1697, 0.4311, 0.5673, 0.5980],
        [0.2870, 0.0539, 0.2289, 0.3007, 0.6262],
        [0.0553, 0.8031, 0.9282, 0.0542, 0.3415],
        [0.7486, 0.5111, 0.2467, 0.0990, 0.6030],
        [0.2950, 0.8395, 0.8924, 0.7440, 0.5326]])
```
We need to do this window operation on every pixel on this image, so we need to add padding to this image the same size as the `radius`.
Doing this gives us the following first and second window for `radius = 1`:
```python
tensor([[0, 0, 0],
        [0, 0.3337,  0.1697],
        [0, 0.2870, 0.0539]])

tensor([[0, 0, 0],
        [0.3337, 0.1697, 0.4311],
        [0.2870, 0.0539, 0.2289]])
```
We have no definite answers on what values are best to use for the padding. The assumption we have right now is that it does not matter, but we did not have enough time to test and prove this assumption.

For each of these windows we create two seperate windows, one containing only the center values(`c_values_window`) and the second one consisting of relative euclidian distances(`distance_weights_windows`). 

The `c_values_window` window is used to calculate the distance for each element compared to the center in the original window, this is done by subtraction.

For the location distance we can create a window filled with relative distances. And remove the values farther away than the `radius` since the kernel will be a square.

The 2 matrices, `c_values_window` and `distance_weights_windows`, will look like this. The second matrix will have the corners set to 0 since this kernel is created with a radius of `1`
```python
tensor([[0.3337, 0.3337, 0.3337],
        [0.3337, 0.3337, 0.3337],
        [0.3337, 0.3337, 0.3337]])

tensor([[1.43 , 1, 1.43],
        [1    , 0, 1   ],
        [1.43 , 1, 1.43]])
```

Now we have an image split up in X amount of pixel value windows, depending on the image size and the radius used. We can similary do the same for the output of the encoder. Create the same X amount of windows.

We can sum all these windows and create a matrix with the same shape as the original image. Where each (i,j) location contains weights summed up for that (i,j) centered window. Now we just do an element wise multiplication between the layer of the encoder output and the newly generated matrix. This method works great, we can even do this batchwise across multiple images and we do this very efficiently, and this stays true to the paper.

## Post-processing

Since the output of the encoder shows the hidden representation, but is still rough. The postprocessing algorithm can be found in the image under Algorithm 2.
We apply postprocessing to the encoder output. A fully conditional random field is applied to smooth the encoder output and provide a more simple segmentation with larger segments. Furthermore, we see that the probability boundary is computed with the contour2ucm method. We have not implemented this postprocessing step as was discussed with the supervisor for this paper. 

## Results

The models were trained on a Google Cloud compute engine running in zone us-west-1b. The machine was initialized with CUDA support and running PyTorch 1.8. The specifications of the machine state the N1 high-memory 2 vCPU’s with 13GB RAM, and it was additionally beefed up with a NVIDIA Tesla K80 sporting 24GB RAM.

## Benchmarking

The postprocessed output of the encoder will be benchmarked against the groundtruth of the BSD300 and BSD500 dataset. The benchmarks will include Segmentation Covering, Variation of information and Probabilistic Rand Index. These were not thoroughly explained in the paper and there is additional rescaling and segmentation clustering involved. 

## Reproduction Discussion

Both of us had around 100 dollars to spend on google cloud for training our models. Since we were novice in cloud computing, of course some of the budget was wasted on models that resulted in nothing useful. But for the most we found that training a model with 400 epochs, 40 batches and a batch size of 5 took around 10 hours to train and cost around 8-10 dollars. Given the fact that the original authors trained the model for 50000 epochs on the full 12000 image dataset, achieving the same result is infeasible as it would never fit in our budget and time for the project. That said, one could cut costs by having their own dedicated hardware, with the expenses we have made you can buy a nice GPU to train your model on (assuming you can get a GPU at MSRP during this semiconductor shortage).


## Conclusion

We have demonstrated a reproduction of the paper "W-net: A Deep Model for Fully Unsupervised Image Segmentation.". We have explained the dataset, the model, losses and postprocessing. Furthermore, we have demonstrated our preliminary results and explained our benchmarking approach of the project. 

We believe the code is complete and true to the paper, but we didn't get the chance to run it in the same way as the paper.



### References
[^x1]: T. (2018, October 17). taoroalin/WNet. GitHub. https://github.com/taoroalin/WNet
[^x2]: G. (2019, November 26). gr-b/W-Net-Pytorch. GitHub. https://github.com/gr-b/W-Net-Pytorch
[^x3]: F. (2019a, June 13). fkodom/wnet-unsupervised-image-segmentation. GitHub. https://github.com/fkodom/wnet-unsupervised-image-segmentation