# page-recognition
Recognize book pages and play a related given sound

Page recognition is actually performed using a given set of reference images, obtained from a flier pages by means of a specific transformation and resize, in order to get a 500px height.

Matches are performed between a video frame acquired by a cam and all the reference images, returning the best one.

A given music is played when a given page is recognized, using a libvlc API python binding.
For this reason, VLC must be installed.

In order to avoid sound interruption due to a temporal jitter in the match, a sort of temporal stabilization is added.
Better approaches will be exploited in future.


Music: http://www.bensound.com