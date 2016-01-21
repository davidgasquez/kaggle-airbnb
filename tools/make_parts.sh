#! /bin/sh

tar -cvvzf airbnb.tar.gz airbnb-kaggle/
split -b 10M airbnb.tar.gz airbnb_
