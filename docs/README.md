# Introduction
What if we could generate infinite data? Can we train any machine learning model then? In a classic document titled ["The Unreasonable Effectiveness of Data"](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf), AI legend Peter Norvig and colleagues suggest that data might be able to solve all problems that involve human beings. While physics can explain the real world; humans remain an enigma that only the promise of infinite data might be able to solve (allegedly).

Using [Conditional Generative Adversarial Networks](https://arxiv.org/pdf/1411.1784.pdf) with [Unreal Engine](https://www.unrealengine.com) there is a way to at least partially answer this question. That is the topic of contention in this project. We explore the limits of roughly the following process:
1. Generate data using Unreal Engine
2. Translate this data into real world data using Conditional Adversarial Networks (e.g. [Image-To-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf))
3. Train a deep neural network on this generated data

Is there a limit to how much data we can generate? What are the challenges? How can we solve them? 

## Programmable Generative Model for Synthetic Data: Unreal Engine
The Unreal Engine provides us with a programmable environment. For programming this environment, we develop a very simple library called [unreal-cv-data](https://github.com/abhay-venkatesh/unreal-cv-data) that is built on [UnrealCV](https://github.com/unrealcv). Using this library is very simple: 
1. download Unreal Engine 
2. install the library, and 
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


### Classes
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
