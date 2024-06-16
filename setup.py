from setuptools import find_packages,setup


def install_req(req_path:str)->list[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(req_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    
    return requirements

setup(
    name='metabolic-syndrome-complete-mlops-pipeline-project',
    author='Ayush K T',
    author_email='thakur8ayush@gmail.com',
    packages=find_packages(),
    install_requires=install_req('requirements.txt')
)