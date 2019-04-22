# Introduction
What if we could generate infinite data? Can we train any machine learning model then? In a classic document titled ["The Unreasonable Effectiveness of Data"](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf), AI legend Peter Norvig and colleagues suggest that data might be able to solve all problems that involve human beings. While physics can explain the real world; humans remain an enigma that only the promise of infinite data might be able to solve (allegedly).

Using [Conditional Generative Adversarial Networks](https://arxiv.org/pdf/1411.1784.pdf) with [Unreal Engine](https://www.unrealengine.com) there is a way to at least partially answer this question. That is the topic of contention in this project. We explore the limits of roughly the following process:
1. Generate data using Unreal Engine
2. Translate this data into real world data using Conditional Adversarial Networks (e.g. [Image-To-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf))
3. Train a deep neural network on this generated data

Is there a limit to how much data we can generate? What are the challenges? How can we solve them? 

## Approach

We begin by [defining a generative model for synthetic data using the Unreal Engine](#programmable-generative-model-for-synthetic-data-unreal-engine). Let us call the Unreal Engine generative model distributions for images $$\mathbb{P}_{\text{Unreal}}(X)$$, and for their masks, $$\mathbb{P}_{\text{Unreal}}(Y)$$. 

### Image-to-Image Translation
Then, we consider a conditional generative adversarial network $$G: Y\times Z \rightarrow X$$, where $$Z\sim \mathcal{N}(\mu,\sigma)$$ is some random noise vector. This cGAN essentially takes a mask and noise vector, and translates back into a fully formed image. Our idea is to train $$G$$ using $$(Y,X)$$ pairs received from the [ICG dataset](https://www.tugraz.at/index.php?id=22387), which is a real world drone dataset. Here, each $$Y$$ is a mask, and $$X$$ is an image. The cGAN will learn to map masks into real images. 

Once we have $$G$$, we can use it to translate a $$Y\sim\mathbb{P}_{\text{Unreal}}(Y)$$ into a real-world looking image. 

TODO: Add diagram describing our approach.

### Generative Model Diversity

Furthermore, we aim to study the effects of generative model diversity on the extension of a model. Is there ways to improve a generative model such that a supervised classifier can be better learned? Our point of departure is [Image-To-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf). However, we also explore other approaches such as [BicycleGAN](https://junyanz.github.io/BicycleGAN/), [SPADE](https://nvlabs.github.io/SPADE/), and an ensemble of these.

### Classifier Extension
To evaluate our setup, we train a classifier on the real-world [ICG dataset](https://www.tugraz.at/index.php?id=22387), and compare it to the performance on the translated images, as well as investigate whether a model trained on real-world data can be "extended" using synthetic data.

## Programmable Generative Model for Synthetic Data: Unreal Engine
The Unreal Engine provides us with a programmable environment. For programming this environment, we develop a very simple library called [unreal-cv-data](https://github.com/abhay-venkatesh/unreal-cv-data) that is built on [UnrealCV](https://github.com/unrealcv/unrealcv). Using this library is very simple: 
1. download [Unreal Engine](https://www.unrealengine.com)
2. install [UnrealCV](https://github.com/unrealcv/unrealcv) and [unreal-cv-data](https://github.com/abhay-venkatesh/unreal-cv-data), and 
3. interface with it in the following manner:

```Python
def collect(self):
  client.connect()
  if not client.isconnected():
    raise RuntimeError("Could not connect to client. ")

  # First we prepare the Unreal Engine environment by preprocessing it
  PreProcessor(self.environment_folder).preprocess()

  # Then we build our dataset
  Builder(self.environment_folder).build(4000)

  client.disconnect()
```

### Examples
<p float="left">
  <img src="./img/unreal_samples/images/49.png" width="155" />
  <img src="./img/unreal_samples/masks/49.png" width="155" /> 
  <img src="./img/unreal_samples/images/42.png" width="155" />
  <img src="./img/unreal_samples/masks/42.png" width="155" /> 
</p>

<p float="left">
  <img src="./img/unreal_samples/images/36.png" width="155" />
  <img src="./img/unreal_samples/masks/36.png" width="155" /> 
  <img src="./img/unreal_samples/images/261.png" width="155" />
  <img src="./img/unreal_samples/masks/261.png" width="155" /> 
</p>

## Data Engineering
We select the following classes and colors:
```json
{
    "paved-area": "(R=128,G=64,B=128,A=255)",
    "rocks": "(R=48,G=41,B=30,A=255)",
    "pool": "(R=0,G=50,B=89,A=255)",
    "water": "(R=28,G=42,B=168,A=255)",
    "vegetation": "(R=107,G=142,B=35,A=255)",
    "roof": "(R=70,G=70,B=70,A=255)",
    "wall": "(R=102,G=102,B=156,A=255)",
    "fence": "(R=190,G=153,B=153,A=255)",
    "car": "(R=9,G=143,B=150,A=255)",
    "tree": "(R=51,G=51,B=0,A=255)",
    "obstacle": "(R=2,G=135,B=115,A=255)"
}
```

Additionally, we have the class "grass" set to (R=0,G=102,B=0,A=255). Hence, we work with a total of 12 classes. The [ICG dataset](https://www.tugraz.at/index.php?id=22387) comes with 24 classes, 12 of which we "turn off" by [setting](https://github.com/abhay-venkatesh/conditional-drones/tree/master/icg_dataset) them to (R=0,G=0,B=0,A=0).

## Evaluation

### Image-To-Image Translation between Unreal and ICG

#### ICG Baseline
<p float="left">
  <img src="./img/translations/icg/053-inputs.png" width="103" />
  <img src="./img/translations/icg/053-outputs.png" width="103" />
  <img src="./img/translations/icg/053-targets.png" width="103" />
  <img src="./img/translations/icg/106-inputs.png" width="103" />
  <img src="./img/translations/icg/106-outputs.png" width="103" />
  <img src="./img/translations/icg/106-targets.png" width="103" />
</p>
<p float="left">
  <img src="./img/translations/icg/113-inputs.png" width="103" />
  <img src="./img/translations/icg/113-outputs.png" width="103" />
  <img src="./img/translations/icg/113-targets.png" width="103" />
  <img src="./img/translations/icg/126-inputs.png" width="103" />
  <img src="./img/translations/icg/126-outputs.png" width="103" />
  <img src="./img/translations/icg/126-targets.png" width="103" />
</p>

#### Testing on Unreal
<p float="left">
  <img src="./img/translations/unreal/9-inputs.png" width="103" />
  <img src="./img/translations/unreal/9-outputs.png" width="103" />
  <img src="./img/translations/unreal/9-targets.png" width="103" />
  <img src="./img/translations/unreal/12-inputs.png" width="103" />
  <img src="./img/translations/unreal/12-outputs.png" width="103" />
  <img src="./img/translations/unreal/12-targets.png" width="103" />
</p>

<p float="left">
  <img src="./img/translations/unreal/19-inputs.png" width="103" />
  <img src="./img/translations/unreal/19-outputs.png" width="103" />
  <img src="./img/translations/unreal/19-targets.png" width="103" />
  <img src="./img/translations/unreal/34-inputs.png" width="103" />
  <img src="./img/translations/unreal/34-outputs.png" width="103" />
  <img src="./img/translations/unreal/34-targets.png" width="103" />
</p>

We note that we get plausible but not yet perfect results. One of the challenges that remains for us is that we cannot find a perfect 1-1 mapping between classes in the source image, and in the unreal environment. However, this is something that is entirely fixable. The unreal environment can in principle have any class, and it is now known to have [photorealistic imagery](https://www.youtube.com/watch?v=E3LtFrMAvQ4). Our intuition is that some work on the data engineering side would greatly improve our results.

### Stuff Segmentation

We observe that our generative model produces accurate "stuff" as in paved area or grass. Hence, we study stuff segmentation using synthetic data.

## Discussion

## Acknowledgements
