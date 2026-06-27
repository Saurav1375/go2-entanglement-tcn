# Per-window leg-entanglement detector output for the Unitree GO2.
# header.stamp is the message timestamp (the recommended way to carry time in ROS 2).

std_msgs/Header header

bool    entangled          # debounced detection decision (true = entanglement alarm)
float32 confidence         # temperature-calibrated probability of entanglement [0,1]

# Per-leg probability of that leg being entangled [0,1]
float32 fr_probability
float32 fl_probability
float32 rr_probability
float32 rl_probability

# Per-leg physics-grounded severity / intensity [0,1] (0 ~= normal)
float32 fr_intensity
float32 fl_intensity
float32 rr_intensity
float32 rl_intensity

string  alarm_leg          # highest-probability leg when entangled ("" otherwise)
