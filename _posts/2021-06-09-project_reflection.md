---
title: PIC16B-Project Reflection
date: 2021-06-09 19:01:10 -0700
categories: [PIC16B-blog]
tags: [machine-learning, project, reflection]
---
Now that the very first offering of PIC16B at UCLA is over, let's reflect on what we, i.e., [Alice](https://naliph.github.io/) and I, had worked and achieved together for the project, which is one of the most crucial components of the class.

**Project Overview**:
*Colorization* has a variety of applications in recreational and historical context. It transforms how we see and perceive photos which helps us tremendously in visualizing and convey stories, emotion, as well as history. For this project, we implemented a machine learning model that would be used to colorize grayscale images. Our project was prepared and trained on approximately 10,000 images; specifically, we utilize the power of a convolutional neural network combined with a classifier deployed through `TensorFlow` and `Keras` to train the model. For more details of the project, please take a look at our [github repo](https://github.com/tducvu/PIC16B-project).

### Some Accomplishments
First, I am pretty proud to say that after so many (and many) times of trials and errors we successfully implemented a machine learning model which can be easily integrated into a local app for immediate use. Even though the result images are not as vibrant and colorful as we expected them to be, the model overall managed to produce the tone/color palette to be quite convincing, compelling and refreshing in some sense. I have to emphasize here that what our model is capable of doing is rather impressive especially when comparing with some of the other pioneers of this area where they get to train their models with a **HUGE** amount of data and computing power. In addition, the [github repository](https://github.com/tducvu/PIC16B-project) is one of the aspects that fulfills and brings our project idea into fruition. Without a doubt, we're really proud with how it turns out, e.g., organized, professional-looking, and very straightforward as well as concise.

![neat](/images/pic16b-reflection/neat.gif)

Well, a product that works is probably what we all desire initially when we start with any type of project; however, what remains the most significant and valuable is still the process in which we devote ourselves to learn, try out new things, and improve and optimize existing models/algorithms. So what I have learnt from pursuing this project over the past 2 months. Let's just dive right into it.

### Learning Experiences
#### Git & Github Navigation
I've been working with Git and Github before, but this is the first time I've had an opportunity to collaborate with someone else on a project. It's certainly an interesting experience as I've gained more knowledge of the workflow of collaborating on Github (e.g., branch, pull-push, merge, etc). In addition, the more I work with the command line the more I appreciate the convenience of setting aliases for some of the commonly used commands. For example, instead of typing the whole line
```shell
git push origin master
```
to push, we can just set it to something much more concise such as
```bash
alias gp="git push origin master"
```
in our Unix shell; I personally use `zsh`, so it's `~/.zshrc` for me. Now, as one may guess, each time we need to push something to Github, we can just run
```shell
gp
```
and ta-da -- we just saved ourselves some time of typing.
#### Web Development with Heroku
Besides Github, I also had some fond and yet frustrating memories with web development. Again, this is my very first time working with deploying a machine learning model into a cloud platform (webapp), more specifically, [Heroku](https://www.heroku.com/). And I have to admit that I made a wrong choice here; Heroku later on proves to be a very restricted host. It's not recommended for projects which are resource hungry/intensive as it provides us with a very limited slug size, which is basically the compressed size of our application (max 500MB -- just our models by themselves takes more than 300MB storage).

At least, we managed to successfully build the user interface and deploy it on the internet which can be found [here](https://colorizing-pic16b.herokuapp.com).

<img src='/images/pic16b-reflection/colorizedweb.png' width=550>

The webapp is there looking really nice, but unfortunately it's not functional at all. More details on this will be discussed later in the **Limitation & Further Improvements** section. Certainly, this experience has taught me quite a lot on the huge difference one can make (in the long run) when committing to something impulsively versus researching thoroughly and carefully before actually jumping into it.
#### "Hardcore" Machine Learning/Training
I am not sure if I use the word "hardcore" here correctly, but oh well, at least it's the case for me. In short, it took me about 3-4 weeks to fine tune and train the model to get some acceptable results. On a bad day, a model that is trained well over 10 hours on a GPU just decides to rebel on me and turns every image it gets to a very red-ish and ominous color tone. On a more moody and chill day, another model with more than 12 hours of learning experience just gets upset with me and determines to isolate itself from living a colorful life, i.e., grayscale images remain grayscale. Nevertheless, statistically speaking, there should be at least a few days in which I consider to be good, right? Of course, I believe in math and specifically probability, and yes I get to experience such days once in a while. What can I say? It's one of the best feelings in the world when the child I train with only 5-7 hours just outperforms all of her older brothers and sisters. As a parent, I cannot be prouder seeing my kids out there living a creative and colorful life.

So what do I actually want to convey through these stories filled with simile and metaphor? --Machine learning is not an easy feat. First and foremost, it's a time-consuming process. It's usually unrealistic to train some big-scale models in a day or two (or even weeks) and expect it to be fully and perfectly functional. We all have bad and good days in our lives, and it's okay to be frustrated and disappointed with the outcome our model produces. I've learnt that I should never give up on my "child" no matter how ill-behaved it is. Patience is the key. By consistently and meticulously pouring efforts into finding the root of the problem and act on it, I believe that my child would eventually get to live a very vibrant and colorful life as her brothers and sisters.

For those who may wonder whether it would cost a lot to raise so many children, let me be upfront here and tell you that money is indeed one of our main concerns, and this leads to my last (but certainly not least) learning experience from completing this project.

<img src='/images/pic16b-reflection/money.jpg' width=400>

#### Frugality x Machine Learning
Money, the one powerful tool that can aid us with achieving our goal much **faster** and make our lives **easier**. When first starting this project, I never thought that money would be a problem for us as mostly everything is open-source and free nowadays (I am an advocate of this idea and thus a Linux user). And as one may have guessed, it turns out money is a major issue that we faced as we dig deeper into the project. From online coursework to computing power and bandwidth/storage, everything is monetized left and right. At some points, I even question myself whether I made the right decision to pursue this project given the limited resources that we have access to and how expensive it is to spend on something like this considering that we're at a stage of our lives where we have to spend money very wisely and frugally. In the end, we did try our best utilizing all the free resources available on the web and end up buying some necessities which are extremely crucial to the success of the project. To sum up, when it comes to spending money on a project, be frugal and always try to utilize all the free and available resources on the web first, or sacrificing some of the functionality/components of the project and not spending a dime on anything at all. I realize that the choice is completely up to us to make, and we should think and consider things very carefully before making such a decision (please don't be like me and purchasing stuffs on an impulse).
### Limitations & Further Improvements
Looking back at the project proposal that we made two months ago, I am seriously amazed of how ambitious our group was during the time. We initially set up the bar quite high when aiming to train the model with roughly a million images, which in reality, we could afford to do so with only about 10k images. Furthermore, our ultimate goal is to build an interactive and fully functional webapp... We did indeed deploy a webapp on the internet, but it's sadly just for look, and the users cannot really colorize any image with it. However, we did successfully create a Python local app as we proposed which we're pretty proud of. Also, one of the aspects that we hope to achieve with this project is to be able to build a model that can predict and colorize historical photos with believable and "accurate" color tone/palette. Again, our model still has many flaws and needs to be undergone through a lot and lots more of training to be capable of carrying out such a task.

So what can we do to further improve upon this project and take it up a notch?
- The main reason that our model cannot be deployed to Heroku is due to its large size and the time it takes to load the model, and in the process, the memory built up exceeds the allowed slug size which triggers the connection error. Thus, one solution is to optimize our model more aggressively to reduce its size and find additional methods to bring down the overall size of the application. Another workaround for this problem is to move on from Heroku to some more machine learning friendly hosts, though I highly doubt that it would be a completely free of charge kind of solution.
- As mentioned above, optimizing the model is one of the things that we need to focus more on. At the time of writing, our model takes more than a minute to run when deploying in a local app; in essence, the colorizing process is not immediate as we desire, and users would certainly notice the unusually long runtime each time they try to colorize an image. Therefore, we want to carry out more research and testings to optimize the algorithm which eventually helps minimize the execution time and thus enhance the users experience overall.
- The sharpness and vibrancy in each colorized image generated from our model is still very lacking and not really on par with our expectation. So we definitely would love to spend more time to train the model with a significantly larger number of images. Without any doubt, it would be a costly process $$$.

### For Future Me
Frankly speaking, before taking this class with Professor Chodrow, which by the way is one of the best decisions I've ever made since transferring to UCLA, I was a very clueless student who's constantly seeking for research topics to pursue. I am not sure what kind of research I will be devoting myself to at graduate school. And yes, I am so glad that I had a chance to take this class and got my hand on doing this project. This experience enlightens me a bit on my inner passion and interest. Well, what can I say at this point -- Machine learning is cool, and data science is just purely awesome. The knowledge obtained from the class and specifically the project would certainly serve me well in the near future as I will be doing more research on these fields and ultimately decide to commit to them in graduate school. In addition, I think my future self would appreciate and benefit a lot from this experience as it has taught me so many things pertinent to my interest.
- I got my first hand-on experience of how to collaborate on Github and build a fun and formal project together with a partner -- thank you Alice for all of your help with the project throughout the quarter. This first project would hugely inspire me and add significant momentum for the creation of the second, third, and more awesome works in the near future.
- Machine learning is a extraordinarily hard, challenging, and yet rewarding field. This experience has deepened my interest in it, and I am so elated that I can create something helpful and meaningful for somebody out there with machine learning.
- This also solidifies my intention of pursuing graduate school and work in the industry (data science/machine learning related of course) later on in my life.

For myself who will be reading this maybe 3 or 5 years from now (06/09/2021), just remember that you worked hard and achieved something remarkable despite the extreme and unusual circumstance... Be proud of who you are and don't even bother what other people say. And remember that it's okay to not be okay; we all have one of those days in our lives. Things will get better in one way or the other. **BELIEVE IN THE PROBABILITY :D**.

Now, without any further ado, let's jump on board and get onto the next exciting part of our journey.

<img src='/images/pic16b-reflection/sailing.gif' width=660>
