import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

path = './env_results/LSTM_New_CorrH_11F_timeF_Unscaled_240Seq_0Label_16Batch_1e-06LR_256Hidden_2LayerDim/'
# Load your PDF files - replace with your actual file paths
pdf1 = path + 'heatmap_no_base_1440EL_Sep 01 2021_to_Sep 01 2022_hor' + '.pdf'
pdf2 = path + 'heatmap_base_1440EL_Sep 01 2021_to_Sep 01 2022_hor' + '.pdf'

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Assuming you are reading the PDFs as images
# You might need to adjust this part depending on how you read your PDFs
axs[0].imshow(plt.imread(pdf1))
axs[0].set_title('(a)', y=-0.2)
axs[0].axis('off')

axs[1].imshow(plt.imread(pdf2))
axs[1].set_title('(b)', y=-0.2)
axs[1].axis('off')

# Save the combined figure
plt.savefig(path + 'combined_heatmap_1440EL_Sep 01 2021_to_Sep 01 2022_hor' + '.pdf', bbox_inches='tight')