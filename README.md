# SOAP-Self-supervised-Online-Adversarial-Purification

Deep neural networks are known to be vulnerable to adversarial examples, where a perturbation in the input space leads to an amplified shift in the latent network representation.
In this paper, we combine canonical supervised learning with self-supervised representation learning, and present Self-supervised Online Adversarial Purification (SOAP), a novel defense strategy that uses a self-supervised loss to purify adversarial examples at test-time. Our approach leverages the label-independent nature of self-supervised signals, and counters the adversarial perturbation with respect to the self-supervised tasks.
SOAP yields competitive robust accuracy against state-of-the-art adversarial training and purification methods, with considerably less training complexity.
In addition, our approach is robust even when adversaries are given knowledge of the purification defense strategy.
To the best of our knowledge, our paper is the first that generalizes the idea of using self-supervised signals to perform online test-time purification.
This paper is accepted to ICLR2021.
