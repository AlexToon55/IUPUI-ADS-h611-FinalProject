# IUPUI-ADS-h611-FinalProject
# Draft of the write up will be here!

# FOR THE TA
I was planning on writing a write up here and then learning how to make animated videos the same way as 3Blue1Brown, all in python. 
However.... That was over ambitious (but still very possible! - just couldn't fit it in the time). I will be moving back to HackMD for the first and I suppose final submission - but hope this explains briefly why i picked GitHub for now but I didn't want to have to send you another URL. 


# Backpropagation by Alex Toon

**"From bronze league to grandmaster, all it takes is back-propagation."**


## Mission 0 - What game are we playing?

Imagine you're playing a competitive game for the 67th time that day. Every attempt you change something, use new a new character/race, try a new strategy, use different units. You want to optimize your build and strategy to do the most damage possible and win the game

In this paper, I aim to explain back-propagation through the lens of a video game Real time strategy (RTS) optimization. For those non gamers, imagine a very intense, live game of chess. Back-propagation is the algorithm that allows neural networks to learn from their mistakes and improve over time.

The literal game I will focus on is Starcraft II, a popular RTS game where players gather resources, build bases, and command armies to defeat their opponents. The goal is to optimize your build order and strategy to maximize your chances of winning. 

**My goal in this write up is to make back-propagation understandable using a small Starcraft 2 example, not to explain every detail of Starcraft or deep learning.**

The key points are:

- **It is real time** - meaning you have to make decisions quickly and adapt to changing situations.
- **Partially observable** - you can't see everything your opponent is doing, so you have to make educated guesses.
- **Long horizon strategy** - decisions made early in the game can have a significant impact on the outcome and games can last up to around 40 minutes and longer. 

In this write up I am going to ignore most of that complexity and focus on one more ingredient **"back-propagation"** . I will use a simplified StarCraft 2 example so we can all see how the math actually works. 

For an example of how advanced this can get, "AlphaStar" was a real example of an AI that mastered Starcraft II using deep reinforcement learning and back-propagation. It learned from millions of games, improving its strategies and decision-making over time and inspired this write up. 
I personally never got to play against it, but i guarantee it would have been a tough match and i got to grandmaster rank (Top 4% of players) in SC2 back in the day.

I hope you enjoy this write up and learn something new about back-propagation and neural networks!

Mission 1 introduces the network structure, Mission 2 covers the forward pass and cost function, Mission 3 explains the chain rule and error measurement, Mission 4 discusses weights and pseudo code, and Mission 5 ties everything together with a recap and real-world applications.

## Mission 1 - The Network as a build order

toy model 

SC2 concepts 


## Mission 2 - The battle: Forward pass and cost

forward pass

cost function 

## Mission 3 - Replay Analysis: chain rule & errors

chain rule 

measuring error

DIAGRAM

## Mission 4 - Rethinking your build - Weights & psueduo code

deltas 

gradients

gradient descent

psuedocode


## Mission 5 - From getting GG'ed to Grandmaster - Putting it all together

### Recap


### From our toy network to real systems - AlphaStar


Back-propagation is the engine behind the success of deep learning, enabling neural networks to learn from their mistakes and improve over time. By understanding the mechanics of back-propagation through the lens of a real-time strategy game like Starcraft II, we can appreciate how this algorithm allows AI to optimize strategies and make better decisions. Whether you're a gamer or a machine learning enthusiast, the principles of back-propagation are fundamental to the advancement of artificial intelligence.

Our toy SC2 win predictor is tiny compared to AlphaStar, but the core story is the same.  Play games, measure error, push blame backwards through the network and slowly, painfully stop getting cheesed, GG'ed and eventually reach Grandmaster level play.

## References

- https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/ 