# GENERATIVE ADVERSAIAL NETWORKS
As part of my research into data generation for the dynamic analysis of materials, I had the opportunity to carry out several implementations of deep learning models using the Pytorch module and following the GENERATUVE ADVERSARIAL NETWORKS (GAN) method. The aim of these models was to produce Frequency Response Function (FRF) data from the excitation coordinates. The three models used and presented in this repository are: a Linear GAN, a Convolutional GAN and a Wasserstein GAN.

## Table of Content
  * [Dataset](#Dataset)
  * [Implementations](#Implementations)
    + [Linear GAN](#Linear GAN)
    + [Convolutionnal GAN](#Convolutionnal GAN)
    + [Wasserstein GAN](#Wasserstein GAN)
  * [Results](#Results)

##  Dataset
The Python code presented in this section corresponds to the creation of the dataset that will be used to train the various models. This dataset consists of an excel file "coordinates.xlsx" containing the various coordinates of the excitation points and the name of the file containing the corresponding FRF data, and a folder containing all the excel files containing the FRF data. In total, this dataset contains 300 FRF data points and their coordinates. 
The "FRFDataset" class transforms the FRF data into a tensor and associates it with the coordinates to create the dataset. The "dataloader" function provided by Pytorch is then used to load the data.

## Implementations
The purpose of a Generative Adversarial Network (GAN) is to learn from samples of real data and then produce its own creations that are indistinguishable from real images without human intervention. To attain this objective, two opposing neural networks are used.

The initial neural network is known as a Generator, and its purpose is to create a fake. The network is given data and uses this to create its own output. For example, if a set of photos of people is fed into the generator, it will create a new photo based on the input. It relies on shared features of the given photo set to generate a photo that is similar. Therefore, it is not an exact copy, but in our case, it is a photograph of a non-existent person.

The input data and generator-managed data are then sent to the concurrent network. The second network is called the \textit{'Discriminator'} and its purpose is to process received data and identify its authenticity. Data will be labelled as false if it deviates widely from the model or if it seems too perfect. The discriminator can discern whether a given data appears natural or not.

The two networks are continually in conflict. If the discriminator network detects erroneous data, it sends it back to the generator network. In this instance, the generator network is still learning and has not yet achieved full proficiency. Simultaneously, the discriminator network has acquired knowledge. Since the two neural systems learn from each other, a profound learning system is established. The generator network aims to create datasets that resemble real data so closely that they cannot be distinguished from genuine examples by the discriminator network. Conversely, the discriminator network aims to comprehensively analyze and understand genuine examples to the extent that forged examples are practically impossible to classify as genuine.

#Linear GAN
In this case, the generator and discriminator have been implemented with linear layers only. This is the most basic implementation of GAN, it is not very complex or deep but can give good results on data that is not too complex.

#Convolutionnal GAN
Here, the linear layers are replaced by convolution layers with activation (LeakyReLU) and normalisation (Batchnorm) layers. This type of implementation is the most promising for complex data in the context of traditional GANs. The generator and discriminator each have 3 convolution layers, which is quite large, so it requires a fairly low learning rate and a fairly large batch size to obtain error-free results.

#Wasserstein GAN
The last implementation is a Generative Adversarial Network model based on Wasserstein theory training (the scientific paper was published in 2017 introducing this new algorithm \cite{ref10}, known as WGAN is reputed to manage considerably more intricate data than traditional models. These models mark a significant advance in GAN training by resolving problems that traditional GANS are unable to solve, such as mode collapse or instability. However, this implementation remains highly intricate and necessitates meticulous selection of hyperparameters and training to achieve optimal performance.

#Results
Finally, the results are illustrated in this place, for the three implementations. The results consist in the result produced by the generator compared to the real data expected, and the tracking of the losses for both generator and discriminator.
