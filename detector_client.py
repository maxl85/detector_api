import requests


server_url = "http://localhost:8000"

class DetectorClient:
    def __init__(self):
        self.server_url = server_url
    
    def run_predict(self, detector_name: str, image: str):
        """
        Распознать изображение с помощью выбранного детектора

        Parameters
        ----------
        detector_name : str
            Имя детектора.
        image : str
            Изображение в формате base64.

        Returns
        -------
        str
            Ответ детектора в формате JSON.
        """
        
        url = f"{self.server_url}/run_predict/"
        response = requests.post(url, json={
            "detector_name": detector_name,
            "image": image
            })
        if response.status_code != 200:
            raise Exception(f"Error from server: {response.json().get('detail', 'Unknown error')}")
        
        return response.json()
    
    def run_train(self, detector_name: str, dataset_path: str):
        """
        Запустить функцию обучения выбранного детектора.

        Parameters
        ----------
        detector_name : str
            Имя детектора.
        dataset_path : str
            Пусть к датасету.

        Returns
        -------
        str
            Ответ детектора в формате JSON.
        """
        
        url = f"{self.server_url}/run_train/"
        response = requests.post(url, json={
            "detector_name": detector_name,
            "dataset_path": dataset_path
            })
        if response.status_code != 200:
            raise Exception(f"Error from server: {response.json().get('detail', 'Unknown error')}")
        
        return response.json()
    
    def get_list(self):
        """
        Получить список всех доступных детекторов
        
        Returns
        -------
        str
            Список в формате JSON.
        """
        
        url = f"{self.server_url}/get_list/"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Error from server: {response.json().get('detail', 'Unknown error')}")
        
        return response.json()
    
    def get_metadata(self, detector_name: str):
        """
        Получить метаданные детектора

        Parameters
        ----------
        detector_name : str
            Имя детектора.

        Returns
        -------
        str
            Метаданные детектора формате JSON.
        """
        
        url = f"{self.server_url}/get_metadata/"
        response = requests.post(url, json={"detector_name": detector_name})
        if response.status_code != 200:
            raise Exception(f"Error from server: {response.json().get('detail', 'Unknown error')}")
        
        return response.json()
