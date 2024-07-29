import numpy as np
import pandas as pd
from TwoParamNoiseGen import genNoise


def gen_synth_data(fs:int = 60, duration:float=3600, noise_mag=1.5, noise_slope = 0.35, save_dir:str = "/home/martin/Documents/Exjobb/eed/.data/raw/train_data/synthetic_data/"):
    max_angle = 35
    slope = 4

    n_samples = int(duration * fs)
    
    timestamps = np.around(np.arange(0, duration, 1/fs),7)
    x= np.zeros(n_samples)
    y= np.zeros(n_samples)

    # create saccades; uses sigmoid function
    _x_saccade = np.linspace(-5, 5, 100)
    saccade = 1/(1 + np.exp(-_x_saccade))



    cur_index = 0
    cur_x = 0
    cur_y = 0
    evt = np.ones(n_samples, dtype=int)
    for i in range(int(duration/2)):
        saccade_sample_length_x = np.random.randint(1, 4)
        saccade_sample_length_y = np.random.randint(1, 4)

        saccade_X = np.diff(saccade[::((100//saccade_sample_length_x)+1)] * saccade_sample_length_x*slope, prepend=0)
        saccade_Y = np.diff(saccade[::((100//saccade_sample_length_y)+1)] * saccade_sample_length_y*slope, prepend=0)


        cur_index += np.random.randint(60, 180)

        if abs(cur_x) > max_angle:
            saccade_X = -1*np.sign(cur_x)*saccade_X
        elif np.random.rand() < 0.5:
            saccade_X = -saccade_X

        if abs(cur_y) > max_angle:
            saccade_Y = -1*np.sign(cur_y)*saccade_Y
        elif np.random.rand() < 0.5:
            saccade_Y = -saccade_Y



        

        x[cur_index:(cur_index+saccade_sample_length_x)] += saccade_X
        y[cur_index:(cur_index+saccade_sample_length_y)] += saccade_Y
        evt[cur_index:(cur_index+saccade_sample_length_x)] = 2
        evt[cur_index:(cur_index+saccade_sample_length_y)] = 2

        cur_x = np.sum(x)
        cur_y = np.sum(y)



    x = np.cumsum(x)
    y = np.cumsum(y)

    magnitude_target = noise_mag
    noise = genNoise.genData(len(x)+2, noise_slope)#genNoise.genNoise(200, 1.25,1.5)#¤len(x)+2
    RMS, STD = genNoise.calcRMSSTD(noise)
    noise = noise* magnitude_target/np.hypot(RMS,STD)
    RMS, STD = genNoise.calcRMSSTD(noise)
    magnitude = np.sqrt(RMS**2 + STD**2)

    print("Properties of generated noise:")
    print(f"RMS: {RMS:.2f}")
    print(f"STD: {STD:.2f}")
    print(f"RMS/STD: {RMS/STD:.2f}")
    print(f"Magnitude: {magnitude:.2f}")
    # add random noise
    #noise_x = np.random.randn(n_samples) * 0.5
    #noise_y = np.random.randn(n_samples) * 0.5

    etdata = pd.DataFrame(
        {
            "t": timestamps,
            "x": x + noise[0],
            "y": y + noise[1],
            "status": np.ones(n_samples, dtype=bool),
            "evt": evt
        }
    )

    name_tag = "synthetic_data_"+ str(fs)+ "hz_" + str(round(duration))+"seconds"+"_noise_type_"+str(round(RMS/STD,2))+"_magnitude_"+str(round(magnitude,2))+".csv"
    print("Synthetic data generated!")

    return etdata, name_tag

if __name__ == "__main__":
    save_dir = "/home/martin/Documents/Exjobb/eed/.data/raw/train_data/synthetic_data/"
    etdata, name_tag = gen_synth_data(duration=3500)
    etdata.to_csv(save_dir + name_tag,index=False)

    # plot etdata
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=etdata["t"][:500], y=etdata["x"][:500], name="x", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=etdata["t"][:500], y=etdata["y"][:500], name="y", mode="lines+markers"))
    fig.update_layout(xaxis_title="Time (s)")
    fig.update_layout(yaxis_title="Angle (°)")

    fig.show()
    



