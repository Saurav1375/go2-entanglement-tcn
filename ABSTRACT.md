# Proprioceptive Leg-Entanglement Detection in Quadrupeds using Multi-Task Causal TCNs

**Problem.** Four-legged robots increasingly work near cables, nets, and people, where a leg can
easily get caught. The trouble is that this is hard to spot. A camera rarely picks out a thin wire, and
the robot's own body usually hides the leg anyway. The robot can feel a caught leg through its joints,
but that signal is faint, and it looks a lot like the robot just standing still. If nothing catches the
problem in time, the robot keeps trying to walk against the snag and can trip or fall.

**Previous approach.** The closest earlier work [1] spots a snag from the robot's own sensors, using a
momentum-based estimate of outside forces, and responds by re-planning the caught leg's next step. It
works, but it only gives a yes-or-no answer. It cannot say which leg is caught or how badly, and its
fixed cut-off tends to miss the small, drawn-out signals that most real snags produce.

**Our approach.** We use the sensors the robot already has: each joint's angle, speed, and torque, the
foot-contact forces, and the body's motion sensor. A short 0.4-second slice of these signals goes into
a small neural network, a causal TCN [2], [3], which answers three questions at every instant: *is a
leg caught, which leg, and how bad is it*. "Causal" simply means the network only looks at the past and
present, never the future, so the model we check offline behaves the same way once it is on the robot.
Each decision takes about a millisecond, which is fast enough to react right away.

Two things make our method new. First, it points to the exact leg that is caught (front or back, left
or right), not just the fact that something is caught. That is what you actually need in order to help
the right leg. Second, it reports how bad the snag is, on a 0-to-100% scale. We had no labelled
"severity" data to learn from, so we work it out from simple physics: how far the leg's motion has
drifted from normal walking, and how hard the leg is pushing while it is barely moving. We also add a
few hand-built signals for the clearest sign of a snag, which is high force with almost no motion.
Together these let the network tell a real snag apart from a robot that is only standing or holding a
stiff pose.

We tested the detector carefully, always on whole recordings it had never seen before. It reaches an F1
score of 0.807 ± 0.142 and names the correct leg about 91% of the time, well ahead of a simple
force-threshold rule, while staying small enough to run on the robot itself. The result is a clear,
real-time answer to three questions a robot can act on: is a leg caught, which one, and how badly.

---

**References**

[1] J. K. Yim, J. Ren, D. Ologan, S. Garcia Gonzalez, and A. M. Johnson, "Proprioception and Reaction
for Walking Among Entanglements," in *Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems
(IROS)*, 2023. arXiv:2304.02129.

[2] S. Bai, J. Z. Kolter, and V. Koltun, "An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling," *arXiv:1803.01271*, 2018.

[3] A. van den Oord *et al.*, "WaveNet: A Generative Model for Raw Audio," in *9th ISCA Speech
Synthesis Workshop*, 2016.
