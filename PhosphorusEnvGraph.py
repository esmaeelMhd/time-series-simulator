import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

#%%
def date2num(date):
    date = pd.to_datetime(date)
    time = date.strftime('%H:%M')
    time = pd.to_datetime(time, format = '%H:%M')
    # time = mdates.datestr2num(time)
    return time

#%%
class PhosphorusEnvGraph:
    """A visualization using matplotlib made to render OpenAI gym environments"""
    def __init__(self, df, title=None):
      self.df = df
      self.p_amounts = np.zeros(len(self.df.index))
      
      # Create a figure on screen and set the title
      fig = plt.figure()
      fig.suptitle(title)
      
      # Create top subplot for P-amounts
      self.p_amount_ax = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)
    
      # Create bottom subplot for Metal amounts
      self.metal_ax = plt.subplot2grid((6, 1), (2, 0), rowspan=8, colspan=1,
                                       sharex=self.p_amount_ax)
      
      # Create a new axis for reward which shares its x-axis with action (Metal)
      self.volume_ax = self.metal_ax.twinx()
      
      # Add padding to make graph easier to view
      plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)
      
      # Show the graph without blocking the rest of the program
      plt.show(block=False)
    
    def _render_phosphorus(self, current_step, p_amounts, real_p_amounts, step_range, dates):
        # Clear the frame rendered last step
        if self.p_amount_ax is not None:
            self.p_amount_ax.clear()
        p_amounts = np.array(p_amounts).reshape(np.array(p_amounts).shape[0])
        real_p_amounts = np.array(real_p_amounts).reshape(np.array(real_p_amounts).shape[0])
        p_amounts = p_amounts[step_range]
        print('p amounts: ', p_amounts[step_range].shape)
        # Plot p amounts
        self.p_amount_ax.plot_date(dates, p_amounts[step_range], '-', label='Predicted-P')
        self.p_amount_ax.plot_date(dates, real_p_amounts[step_range], '-', label='Real-P')

        # Show legend, which uses the label we defined for the plot above
        self.p_amount_ax.legend()
        legend = self.p_amount_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)
        last_date = date2num(self.df.index.values[current_step-1])
        last_p_amount = p_amounts[current_step-1]
        
        # Annotate the current P-amount
        self.p_amount_ax.annotate('{0:.2f}'.format(last_p_amount),     
          (last_date, last_p_amount),
          xytext=(last_date, last_p_amount),
          bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
          color="black",
          fontsize="small")
        
        # Add space above and below min/max P-amount
        self.p_amount_ax.set_ylim(
          min(self.p_amounts[np.nonzero(self.p_amounts)]) / 1.25,    
          max(self.p_amounts) * 1.25)
    
    def _render_metal(self, current_step, metal_amounts, dates, step_range):
        self.metal_ax.clear()
        # Plot price using candlestick graph from mpl_finance
        metal_amounts = np.array(metal_amounts).reshape(np.array(metal_amounts).shape[0], -1)
        self.metal_ax.plot_date(dates, metal_amounts[step_range], '-', label='Metal Amounts')
        
        # Show legend, which uses the label we defined for the plot above
        self.metal_ax.legend()
        legend = self.metal_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)    
        last_date = date2num(self.df.index.values[current_step-1])
        last_metal_amount = metal_amounts[current_step-1]
        
        # Print the current price to the price axis
        self.metal_ax.annotate('{0:.2f}'.format(last_metal_amount),
          (last_date, last_metal_amount),
          xytext=(last_date, last_metal_amount),
          bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
          color="black",
          fontsize="small")
        
        # Add space above and below min/max metal amounts
        self.metal_ax.set_ylim(
          min(metal_amounts[np.nonzero(metal_amounts)]) / 1.25,    
          max(metal_amounts) * 1.25)
        
    def render(self, current_step, p_amounts, real_p_amounts, metal_amounts, window_size=40):
        # Defne window range (number of time steps per each window)
        window_start = max(current_step - window_size -1, 0)
        step_range = range(window_start, current_step)
        print(step_range)
        
        # Format dates as timestamps, necessary for the graph
        dates = np.array([date2num(x) for x in self.df.index.values[step_range]])
        dates = dates.reshape(dates.shape[0])
        print('dates: ', dates.shape)
        
        # Render Phosphorus amounts (predicted and real)
        self._render_phosphorus(current_step, p_amounts, real_p_amounts, window_size, dates)
        # Render Metal amounts
        self._render_metal(current_step, metal_amounts, dates, step_range)
        
        # self._render_volume(self.round, self.obs, dates, step_range)
        # self._render_trades(self.round, trades, step_range)
        
        # Format the date ticks to be more easily read
        self.metal_ax.set_xticklabels(self.df.index.values[step_range], 
                                      rotation=45, horizontalalignment='right')
        
        # Hide duplicate date labels
        plt.setp(self.p_amount_ax.get_xticklabels(), visible=False)
        
        # Necessary to view frames before they are unrendered    
        plt.pause(0.001)
        
    def close(self):
        plt.close()