import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


filename = 'politechnikagdanska.mp3' 

def generate_custom_spectrogram(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        if len(y) > 2 * sr:
            y = y[:2*sr]

        n_fft = 2048
        hop_length = 512
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))

        img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, 
                                       x_axis='time', y_axis='linear', 
                                       ax=ax, cmap='inferno',
                                       vmin=-100, vmax=0)

        ax.set_ylim(0, 10000)
        
        def khz_formatter(x, pos):
            return '{:g}'.format(x/1000)
        
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(khz_formatter))
        ax.set_ylabel('Częstotliwość (kHz)', color='white')
        ax.set_xlabel('Czas (s)', color='white')
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, color='white')

        sec_ax = ax.secondary_yaxis('right', functions=(lambda x: x, lambda x: x))
        sec_ax.yaxis.set_major_formatter(ticker.FuncFormatter(khz_formatter))
        sec_ax.set_ylabel('')
        sec_ax.tick_params(axis='y', colors='white')
        sec_ax_x = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
        sec_ax_x.set_xlabel('')
        sec_ax_x.tick_params(axis='x', colors='white')

        cbar = fig.colorbar(img, ax=ax, format='%+2.0f')
        cbar.ax.set_ylabel('dBFS', rotation=270, labelpad=15, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.outline.set_edgecolor('white')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Wystąpił błąd: {e}")

if __name__ == "__main__":
    generate_custom_spectrogram(filename)