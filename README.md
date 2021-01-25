# Hack RussianSuperGlue

## Welcome to this repository! ##
Here you can find the results of our study where we tried to solve natural language understanding tasks with the help of rule-base heuristics.

### What you can find in this repository: ###
  *	Python notebooks with detailed descriptions of each and every dataset that is on [RussianSuperGlue](https://russiansuperglue.com/) with various graphics and schemes
  *	Baseline for how to solve each dataset if we:
    * Picked a majority class
    * Picked random option
    * Picked random option, were classes balanced   
  * Suspicious cases in annotation, where you either cannon possibly give an answer or the answer is wrong
  * Heuristics specified for every dataset separately with the help of which you can try to solve it and achieve plausible if not comparable with leaderboard models performance

### What you should not expect to find in this repository: ###
  * Data from RussianSuperGlue, since it is loaded to the workspace in the first cell of each notebook. 
  
### What to do with all of these: ###
#### In order to gain access to our findings (and maybe contribute to our study by finding some new info) we recommend following this algorithm: ####
1.	Choose the task you liked below:
    * [Linguistic Diagnostic for Russian](https://colab.research.google.com/github/tatiana-iazykova/2020_HACK_RUSSIANSUPERGLUE/blob/main/RSG_LiDiRus.ipynb)
    * [Russian Commitment Bank](https://colab.research.google.com/github/tatiana-iazykova/2020_HACK_RUSSIANSUPERGLUE/blob/main/RSG_RCB.ipynb)
    * [Choice of Plausible Alternatives for Russian language](https://colab.research.google.com/github/tatiana-iazykova/2020_HACK_RUSSIANSUPERGLUE/blob/main/RSG_PARus.ipynb)
    * [Russian Multi-Sentence Reading Comprehension](https://colab.research.google.com/github/tatiana-iazykova/2020_HACK_RUSSIANSUPERGLUE/blob/main/RSG_MuSeRC.ipynb)
    * [Textual Entailment Recognition for Russian](https://colab.research.google.com/github/tatiana-iazykova/2020_HACK_RUSSIANSUPERGLUE/blob/main/RSG_TERRa.ipynb)
    * [Russian Words in Context (based on RUSSE)](https://colab.research.google.com/github/tatiana-iazykova/2020_HACK_RUSSIANSUPERGLUE/blob/main/RSG_RUSSE.ipynb)
    * [The Winograd Schema Challenge (Russian)](https://colab.research.google.com/github/tatiana-iazykova/2020_HACK_RUSSIANSUPERGLUE/blob/main/RSG_RWSD.ipynb)
    * [Yes/no Question Answering Dataset for the Russian](https://colab.research.google.com/github/tatiana-iazykova/2020_HACK_RUSSIANSUPERGLUE/blob/main/RSG_DaNetQA.ipynb)
    * [Russian Reading Comprehension with Commonsense Reasoning](https://colab.research.google.com/github/tatiana-iazykova/2020_HACK_RUSSIANSUPERGLUE/blob/main/RSG_RuCoS.ipynb)
2.	Explore away
3.	If you found something interested, feel free to create new code/text cell at the beginning of the notebook and tell us about it. If you want to, you can change the code in our cells as well, yet if you want to show it to us, please do so and create the text cell at the beginning of the file with the description what did you do and in which section of the notebook. It will help us to find your contribution faster!

**BEWARE: next step may not work**

4.	When you are finished, click **File** at the Colab menu panel at the upper end of the workspace. And try to choose **Save a copy in Github** with selecting the repository and slightly changing the filename (you can add your nickname or some numbers etc)

https://github.com/tatiana-iazykova/2020_HACK_RUSSIANSUPERGLUE
  
5.	If step 4 doesn’t work, **download** the notebook and send it to one of the emails: 
      * tvyazykova@edu.hse.ru

