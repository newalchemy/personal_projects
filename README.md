# Welcome!

Welcome to my repository!  Here you will find several programming / machine learning projects of mine, including:

 - Implementing Penalized K-Means by Gradient Descent on reduced-dimension images
 - Diagnosing Psychiatric Illness with an LSTM, using a very similar method to one used in NLP Sentiment Analysis
 - Using XGBoost to predict forest fire size
 
 The first two projects are the .inpynb files in the top directory, and the last one is in the forestFire directory.
 
These projects were written as standalone programming assignments instead of software engineering projects.  There's several things
that I would do differently if I was writing these projects inside of an enterprise, including:
 - Making each script configurable with an outside configuration file that's loaded and parsed, so users can test multiple different 
 configurations easily.
 - Not hardcode any parameters, at all.
 - Automatically store generated plots and performance data into sub-directories which also store the passed configuration by copying
 it into there.  Provide different titles to the plots (sometimes based on the configuration), all to ensure that someone looking at  
 the data can quickly determine what configuration was run and what information they are supposed to learn from the plots.
 - Separate helper functions into utility folders instead of keeping them within the main script for running experiments, which makes it 
 easier for teammates to use them if they wish.
 - Provide separate consideration to data loading, storage, and management.  In each of these cases, the data used was either stored 
 online and downloaded with its own API, or stored in .csv files.  If the data were stored in a database(possibly on a separate server), 
 then we would need to think about how the data is stored and the best way to query it, with regards to runtime.  
 If the data were too large to fit into RAM, we'd have to delete and reload different batches of data into memory during each training 
 step (I may go back and do this for the Psychiatric Illness project in the future).
 
If you have any suggestions or contributions, please feel free to message me.  I hope you enjoy looking at this repository.
