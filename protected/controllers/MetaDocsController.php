<?php

class MetaDocsController extends Controller
{
	public $layout='//layouts/responsiveLayout';
	private $_model;

	public function filters()
	{
		return array(
			'accessControl', 
		);
	}

	public function accessRules()
	{
		return array(
			array('allow',  
				'actions'=>array('index','view','TituloSearch'),
				'users'=>array('*'),
			),
			array('allow', 
				'actions'=>array('admin','create'),
				'users'=>array('@'),
			),
			array('allow', 
				'actions'=>array('delete','update'),
				'users'=>array('admin'),
			),
			array('deny', 
				'users'=>array('*'),
			),
		);
	}

	public function actionView()
	{
		$this->render('view',array(
			'model'=>$this->loadModel(),
		));
	}

	public function actionCreate()
	{
		$model=new MetaDocs;

		$this->performAjaxValidation($model);

		if(isset($_POST['MetaDocs']))
		{
			$model->attributes=$_POST['MetaDocs'];
		

			if($model->save())
				$this->redirect(array('view','id'=>$model->id));
		}

		$this->render('create',array(
			'model'=>$model,
		));
	}

	public function actionUpdate()
	{
		$model=$this->loadModel();

		$this->performAjaxValidation($model);

		if(isset($_POST['MetaDocs']))
		{
			$model->attributes=$_POST['MetaDocs'];
		
			if($model->save())
				$this->redirect(array('view','id'=>$model->id));
		}

		$this->render('update',array(
			'model'=>$model,
		));
	}

	public function actionDelete()
	{
		if(Yii::app()->request->isPostRequest)
		{
			$this->loadModel()->delete();

			if(!isset($_GET['ajax']))
				$this->redirect(array('index'));
		}
		else
			throw new CHttpException(400,
					Yii::t('app', 'Invalid request. Please do not repeat this request again.'));
	}

	public function actionIndex()
	{
		$dataProvider=new CActiveDataProvider('MetaDocs');
		$this->render('index',array(
			'dataProvider'=>$dataProvider,
		));
	}

	public function actionAdmin()
	{
		$model=new MetaDocs('search');
		if(isset($_GET['MetaDocs']))
			$model->attributes=$_GET['MetaDocs'];

		$this->render('admin',array(
			'model'=>$model,
		));
	}

	public function loadModel()
	{
		if($this->_model===null)
		{
			if(isset($_GET['id']))
				$this->_model=MetaDocs::model()->findbyPk($_GET['id']);
			if($this->_model===null)
				throw new CHttpException(404, Yii::t('app', 'The requested page does not exist.'));
		}
		return $this->_model;
	}

	protected function performAjaxValidation($model)
	{
		if(isset($_POST['ajax']) && $_POST['ajax']==='meta-docs-form')
		{
			echo CActiveForm::validate($model);
			Yii::app()->end();
		}
	}
                public function actionTituloSearch($term) {
        // TODO: Arreglar para que busque también cuando se tecleé el nombre, si conviene.
        //$sql = 'SELECT docIdent FROM clientes WHERE docIdent LIKE \'%' . trim($term) . '%\' LIMIT 10';
        $sql = 'SELECT distinct titulo as label, titulo as value FROM metaDocs WHERE titulo LIKE :qterm ORDER BY titulo ASC LIMIT 10';
//        $sql = 'SELECT docIdent FROM clientes WHERE docIdent LIKE \'%94%\' LIMIT 10';
        $command = Yii::app()->db->createCommand($sql);
        $qterm = '%'.$_GET['term'].'%';
        $command->bindParam(":qterm", $qterm, PDO::PARAM_STR);
        $result = $command->queryAll();
        echo CJSON::encode($result); 
        exit;
    }
}
