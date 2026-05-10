    plt.imshow(img.detach().numpy(), cmap="gray")
    plt.grid(False)
    plt.axis('off')
plt.savefig("figs/original_sample")

plt.figure()

img=torch.zeros((dim,dim,channel))
for i in range(batch_size):
    for j in range(channel):
        img[:,:,j]=recovered[i,j,:,:]
    plt.subplot(8,int(batch_size/8),i+1)
    plt.imshow(img.detach().numpy(), cmap="gray")
    plt.grid(False)
    plt.axis('off')
plt.savefig("figs/reconstructed_sample")
