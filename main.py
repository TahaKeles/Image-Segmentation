import sys
import kmeans
import numpy as np
from PIL import Image

def main():

    if(len(sys.argv) != 5):
        print("Try again with correct arguments")

    # reading pixels
    print("Reading pixels...", end="\t")
    



    # read image pixels
    # and have a list in 1 X (width * height) dimensions

    im = Image.open(str(sys.argv[1]))

    arr_im = np.array(im)

    x = arr_im.shape[0]
    y = arr_im.shape[1]

    inp = arr_im.reshape( arr_im.shape[0] * arr_im.shape[1], 3 )


    print("DONE")

    
    model = kmeans.KMeans(
        X=np.array(inp),
        n_clusters=int(sys.argv[2]),
        max_iterations=int(sys.argv[3]),
        epsilon=float(sys.argv[4]),
        distance_metric="euclidian"
    )
    print("Fitting...")
    model.fit()    
    print("Fitting... DONE")

    print("Predicting...")
    color1 = (134, 66, 176)
    color2 = (34, 36, 255)
    color3 = (94, 166, 126)
    print(f"Prediction for {color1} is cluster {model.predict(color1)}")
    print(f"Prediction for {color2} is cluster {model.predict(color2)}")
    print(f"Prediction for {color3} is cluster {model.predict(color3)}")

    # replace image pixels with color palette
    # (cluster centers) found in the model

    # save the final image


    for eachX in range(x):
        for eachY in range(y):
            will_predict = im.getpixel((eachY,eachX))
            predicted = model.predict(will_predict)
            color = model.get_cluster_center(predicted)
            im.putpixel((eachY,eachX),(round(color[0]),round(color[1]),round(color[2])))

    im.save(str(sys.argv[1]+"_"+str(sys.argv[2])+"_colors_"+str(sys.argv[3])+"_epochs_epsilon_"+str(sys.argv[4])+".png"))


if __name__ == "__main__":
    main()