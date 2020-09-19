from distutils.core import setup


setup(
  name = 'tuaneda',         # How you named your package folder (MyLib)
  packages = ['tuaneda'],   # Chose the same as "name"
  version = '0.0.8',      
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Quick EDA and important variable analysis',   # Give a short description about your library
  author = 'Tuan Bui',                   
  author_email = 'kevbui.29@gmail.com',      
  url = 'https://github.com/ubunTuan',  
  download_url = 'https://github.com/ubunTuan/tuaneda/archive/0.0.8.tar.gz',    
  keywords = ['eda', 'important variables', 'exploratory analysis'],   
  install_requires=[           
    'seaborn','matplotlib','sklearn','panda',
    'numpy','scipy','statsmodels','missingno','eli5'
      ],  
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6'
  ],
)





