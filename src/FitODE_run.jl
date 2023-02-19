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


# for i in 1:80 calc_pred_loss(3000; ode3_only=true) end
#  ODE3: median = 1.366e+02; mean = 1.324e+02; sd = 1.322e+01
#  ODE3: median = 1.136e+02; mean = 1.110e+02; sd = 1.087e+01
#  ODE3: median = 9.930e+01; mean = 9.373e+01; sd = 1.835e+01
#  ODE3: median = 1.002e+02; mean = 9.426e+01; sd = 1.819e+01
#  ODE3: median = 6.553e+01; mean = 6.578e+01; sd = 2.168e+01
#  ODE3: median = 6.523e+01; mean = 7.123e+01; sd = 1.944e+01
#  ODE3: median = 1.253e+02; mean = 1.194e+02; sd = 2.055e+01
#  ODE3: median = 8.684e+01; mean = 8.529e+01; sd = 1.188e+01
#  ODE3: median = 1.451e+02; mean = 1.410e+02; sd = 1.471e+01
#  ODE3: median = 8.866e+01; mean = 9.017e+01; sd = 2.183e+01
#  ODE3: median = 1.182e+02; mean = 1.150e+02; sd = 3.338e+01
#  ODE3: median = 8.663e+01; mean = 8.418e+01; sd = 1.360e+01
#  ODE3: median = 8.035e+01; mean = 8.361e+01; sd = 2.221e+01
#  ODE3: median = 6.310e+01; mean = 6.667e+01; sd = 1.198e+01
#  ODE3: median = 8.664e+01; mean = 8.779e+01; sd = 2.212e+01
#  ODE3: median = 7.047e+01; mean = 6.708e+01; sd = 1.348e+01
#  ODE3: median = 1.008e+02; mean = 9.840e+01; sd = 2.610e+01
#  ODE3: median = 6.851e+01; mean = 6.788e+01; sd = 2.202e+01
#  ODE3: median = 1.428e+02; mean = 1.389e+02; sd = 1.576e+01
#  ODE3: median = 6.128e+01; mean = 6.650e+01; sd = 2.646e+01
#  ODE3: median = 4.401e+01; mean = 4.946e+01; sd = 1.692e+01
#  ODE3: median = 4.212e+01; mean = 4.908e+01; sd = 1.769e+01
#  ODE3: median = 9.402e+01; mean = 8.881e+01; sd = 1.971e+01
#  ODE3: median = 1.131e+02; mean = 1.095e+02; sd = 1.379e+01
#  ODE3: median = 1.190e+02; mean = 1.146e+02; sd = 3.282e+01
#  ODE3: median = 1.534e+02; mean = 1.497e+02; sd = 1.124e+01
#  ODE3: median = 1.031e+02; mean = 1.014e+02; sd = 1.981e+01
#  ODE3: median = 1.113e+02; mean = 1.078e+02; sd = 1.596e+01
#  ODE3: median = 1.134e+02; mean = 1.107e+02; sd = 1.046e+01
#  ODE3: median = 5.884e+01; mean = 6.497e+01; sd = 1.700e+01
#  ODE3: median = 1.181e+02; mean = 1.147e+02; sd = 2.719e+01
#  ODE3: median = 4.061e+01; mean = 4.721e+01; sd = 1.820e+01
#  ODE3: median = 1.353e+02; mean = 1.326e+02; sd = 1.832e+01
#  ODE3: median = 6.157e+01; mean = 6.080e+01; sd = 1.339e+01
#  ODE3: median = 8.869e+01; mean = 8.520e+01; sd = 1.199e+01
#  ODE3: median = 1.163e+02; mean = 1.131e+02; sd = 2.770e+01
#  ODE3: median = 4.609e+01; mean = 5.240e+01; sd = 1.978e+01
#  ODE3: median = 4.970e+01; mean = 5.561e+01; sd = 2.109e+01
#  ODE3: median = 9.321e+01; mean = 9.468e+01; sd = 3.061e+01
#  ODE3: median = 1.294e+02; mean = 1.240e+02; sd = 3.298e+01
#  ODE3: median = 1.262e+02; mean = 1.205e+02; sd = 1.997e+01
#  ODE3: median = 7.975e+01; mean = 8.124e+01; sd = 1.374e+01
#  ODE3: median = 5.002e+01; mean = 5.213e+01; sd = 8.289e+00
#  ODE3: median = 8.846e+01; mean = 8.613e+01; sd = 1.103e+01
#  ODE3: median = 4.558e+01; mean = 5.251e+01; sd = 1.997e+01
#  ODE3: median = 4.413e+01; mean = 5.095e+01; sd = 1.886e+01
#  ODE3: median = 7.001e+01; mean = 6.669e+01; sd = 1.374e+01
#  ODE3: median = 7.648e+01; mean = 7.471e+01; sd = 2.181e+01
#  ODE3: median = 6.333e+01; mean = 6.186e+01; sd = 1.377e+01
#  ODE3: median = 1.360e+02; mean = 1.294e+02; sd = 2.054e+01
#  ODE3: median = 1.030e+02; mean = 9.748e+01; sd = 1.687e+01
#  ODE3: median = 8.931e+01; mean = 8.700e+01; sd = 8.935e+00
#  ODE3: median = 1.307e+02; mean = 1.251e+02; sd = 2.299e+01
#  ODE3: median = 5.593e+01; mean = 6.221e+01; sd = 1.534e+01
#  ODE3: median = 7.086e+01; mean = 7.587e+01; sd = 2.065e+01
#  ODE3: median = 1.072e+02; mean = 1.059e+02; sd = 2.933e+01
#  ODE3: median = 1.506e+02; mean = 1.471e+02; sd = 1.089e+01
#  ODE3: median = 8.603e+01; mean = 8.401e+01; sd = 7.684e+00
#  ODE3: median = 9.826e+01; mean = 9.758e+01; sd = 2.101e+01
#  ODE3: median = 6.336e+01; mean = 6.769e+01; sd = 1.308e+01
#  ODE3: median = 9.763e+01; mean = 9.220e+01; sd = 1.875e+01
#  ODE3: median = 1.117e+02; mean = 1.086e+02; sd = 1.076e+01
#  ODE3: median = 1.215e+02; mean = 1.164e+02; sd = 2.168e+01
#  ODE3: median = 5.093e+01; mean = 5.456e+01; sd = 9.908e+00
#  ODE3: median = 8.588e+01; mean = 8.827e+01; sd = 2.243e+01
#  ODE3: median = 1.378e+02; mean = 1.340e+02; sd = 1.210e+01
#  ODE3: median = 7.876e+01; mean = 8.090e+01; sd = 1.385e+01
#  ODE3: median = 5.126e+01; mean = 5.483e+01; sd = 1.907e+01
#  ODE3: median = 8.625e+01; mean = 8.505e+01; sd = 1.201e+01
#  ODE3: median = 1.318e+02; mean = 1.259e+02; sd = 1.753e+01
#  ODE3: median = 8.517e+01; mean = 8.305e+01; sd = 8.035e+00
#  ODE3: median = 7.046e+01; mean = 7.372e+01; sd = 3.071e+01
#  ODE3: median = 1.173e+02; mean = 1.137e+02; sd = 3.312e+01
#  ODE3: median = 3.774e+01; mean = 4.147e+01; sd = 1.012e+01
#  ODE3: median = 8.083e+01; mean = 8.323e+01; sd = 3.311e+01
#  ODE3: median = 5.071e+01; mean = 5.421e+01; sd = 9.375e+00
#  ODE3: median = 9.519e+01; mean = 9.713e+01; sd = 3.079e+01
#  ODE3: median = 1.397e+02; mean = 1.359e+02; sd = 1.233e+01
#  ODE3: median = 5.364e+01; mean = 5.873e+01; sd = 1.320e+01
#  ODE3: median = 9.177e+01; mean = 9.151e+01; sd = 2.715e+01
