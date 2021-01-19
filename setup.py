from distutils.core import setup


setup(
  name = 'tuaneda',         # How you named your package folder (MyLib)
  packages = ['tuaneda'],   # Chose the same as "name"
  version = '0.4.0',      
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Data structures, quick EDA, important variable analysis',   # Give a short description about your library
  author = 'Tuan Bui',                   
<<<<<<< HEAD
  author_email = 'dev@tuanab.app',      
  url = 'https://github.com/tuanab',  
  download_url = 'https://github.com/tuanab/tuaneda/archive/0.4.0.tar.gz',    
  keywords = ['eda', 'data structures', 'important variables', 'exploratory analysis'],   
=======
  author_email = 'kevbui.29@gmail.com',      
  url = 'https://github.com/ubunTuan',  
  download_url = 'https://github.com/ubunTuan/tuaneda/archive/v0.2.5.tar.gz',    
  keywords = ['eda', 'important variables', 'exploratory analysis'],   
>>>>>>> 35bcb411d33f1fd135cca11c379fbc1c171bf8e1
  install_requires=[           
    'seaborn','matplotlib','sklearn','panda',
    'numpy','scipy', 'tqdm'
      ],  
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.6'
  ],
)





