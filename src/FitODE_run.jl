# This branch of git code does only one thing, calculating the loss for the prediction
# portion of Figure 4 for time interval [60,90]

using FitODE

calc_pred_loss();

# Sample runs:

# calc_pred_loss(3000);
#  ODE2: median = 9.474e+01; mean = 9.609e+01; sd = 1.517e+01
# NODE2: median = 4.101e+01; mean = 4.296e+01; sd = 6.192e+00
#  ODE3: median = 8.094e+01; mean = 7.812e+01; sd = 2.172e+01
# NODE3: median = 1.772e+02; mean = 1.797e+02; sd = 3.932e+01
#  ODE4: median = 2.756e+04; mean = 2.921e+04; sd = 9.759e+03
# NODE4: median = 1.654e+05; mean = 1.395e+05; sd = 9.213e+04

# calc_pred_loss(3000);
#  ODE2: median = 9.483e+01; mean = 9.597e+01; sd = 1.553e+01
# NODE2: median = 4.092e+01; mean = 4.310e+01; sd = 6.354e+00
#  ODE3: median = 4.746e+01; mean = 5.489e+01; sd = 2.290e+01
# NODE3: median = 1.665e+02; mean = 1.676e+02; sd = 3.817e+01
#  ODE4: median = 2.531e+05; mean = 2.496e+05; sd = 1.599e+04
# NODE4: median = 6.020e+04; mean = 5.760e+04; sd = 8.849e+03

# calc_pred_loss(3000);
#  ODE2: median = 9.478e+01; mean = 9.622e+01; sd = 1.581e+01
# NODE2: median = 4.102e+01; mean = 4.310e+01; sd = 6.406e+00
#  ODE3: median = 1.148e+02; mean = 1.117e+02; sd = 3.256e+01
# NODE3: median = 1.727e+02; mean = 1.736e+02; sd = 3.882e+01
#  ODE4: median = 1.966e+04; mean = 2.176e+04; sd = 1.036e+04
# NODE4: median = 7.996e+04; mean = 1.001e+05; sd = 9.398e+04

# calc_pred_loss(3000);
#  ODE2: median = 9.522e+01; mean = 9.636e+01; sd = 1.557e+01
# NODE2: median = 4.098e+01; mean = 4.305e+01; sd = 6.302e+00
#  ODE3: median = 1.521e+02; mean = 1.490e+02; sd = 1.047e+01
# NODE3: median = 1.721e+02; mean = 1.730e+02; sd = 3.836e+01
#  ODE4: median = 3.358e+05; mean = 3.358e+05; sd = 1.311e+03
# NODE4: median = 1.457e+05; mean = 1.270e+05; sd = 5.173e+04


for i in 1:30 calc_pred_loss(3000; ode3_only=true) end
 ODE3: median = 1.366e+02; mean = 1.324e+02; sd = 1.322e+01
 ODE3: median = 1.136e+02; mean = 1.110e+02; sd = 1.087e+01
 ODE3: median = 9.930e+01; mean = 9.373e+01; sd = 1.835e+01
 ODE3: median = 1.002e+02; mean = 9.426e+01; sd = 1.819e+01
 ODE3: median = 6.553e+01; mean = 6.578e+01; sd = 2.168e+01
 ODE3: median = 6.523e+01; mean = 7.123e+01; sd = 1.944e+01
 ODE3: median = 1.253e+02; mean = 1.194e+02; sd = 2.055e+01
 ODE3: median = 8.684e+01; mean = 8.529e+01; sd = 1.188e+01
 ODE3: median = 1.451e+02; mean = 1.410e+02; sd = 1.471e+01
 ODE3: median = 8.866e+01; mean = 9.017e+01; sd = 2.183e+01
 ODE3: median = 1.182e+02; mean = 1.150e+02; sd = 3.338e+01
 ODE3: median = 8.663e+01; mean = 8.418e+01; sd = 1.360e+01
 ODE3: median = 8.035e+01; mean = 8.361e+01; sd = 2.221e+01
 ODE3: median = 6.310e+01; mean = 6.667e+01; sd = 1.198e+01
 ODE3: median = 8.664e+01; mean = 8.779e+01; sd = 2.212e+01
 ODE3: median = 7.047e+01; mean = 6.708e+01; sd = 1.348e+01
 ODE3: median = 1.008e+02; mean = 9.840e+01; sd = 2.610e+01
 ODE3: median = 6.851e+01; mean = 6.788e+01; sd = 2.202e+01
 ODE3: median = 1.428e+02; mean = 1.389e+02; sd = 1.576e+01
 ODE3: median = 6.128e+01; mean = 6.650e+01; sd = 2.646e+01
 ODE3: median = 4.401e+01; mean = 4.946e+01; sd = 1.692e+01
 ODE3: median = 4.212e+01; mean = 4.908e+01; sd = 1.769e+01
 ODE3: median = 9.402e+01; mean = 8.881e+01; sd = 1.971e+01
 ODE3: median = 1.131e+02; mean = 1.095e+02; sd = 1.379e+01
 ODE3: median = 1.190e+02; mean = 1.146e+02; sd = 3.282e+01
 ODE3: median = 1.534e+02; mean = 1.497e+02; sd = 1.124e+01
 ODE3: median = 1.031e+02; mean = 1.014e+02; sd = 1.981e+01
 ODE3: median = 1.113e+02; mean = 1.078e+02; sd = 1.596e+01
 ODE3: median = 1.134e+02; mean = 1.107e+02; sd = 1.046e+01
 ODE3: median = 5.884e+01; mean = 6.497e+01; sd = 1.700e+01
