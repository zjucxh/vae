# vae
Variatonal AutoEncoder

## TODO
- Load cloth_in_wind dataset(done)
- Modify reconstruction loss to signed distance loss (done)
- Alternate encoder decoder from residual network to resnet18 (Linear layer to convolution for faster training)  ( done )
- given sdf for data 0, evolve shape deformation with level set method, the sdf take vertex as input, out put signed distance 
- the loss of sdf should include signed distance and **PDE constraints**
- Load cmu simulation dataset
- Feed to GRU
- Train with MSE loss to see the preliminary results
- Think about where the template obj file should be loaded