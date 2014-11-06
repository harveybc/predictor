<?php

class Aislamiento_acometidaController extends Controller
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
				'actions'=>array('index','view'),
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
		$model=new Aislamiento_acometida;

		$this->performAjaxValidation($model);

		if(isset($_POST['Aislamiento_acometida']))
		{
			$model->attributes=$_POST['Aislamiento_acometida'];
		

			if($model->save())
				$this->redirect(array('/aislamiento_tierra/view','id'=>$model->Toma));
		}
                
                if (isset($_GET['Toma']))
                    $model->Toma=$_GET['Toma'];
		$this->render('create',array(
			'model'=>$model,
		));
	}

	public function actionUpdate()
	{
		$model=$this->loadModel();

		$this->performAjaxValidation($model);

		if(isset($_POST['Aislamiento_acometida']))
		{
			$model->attributes=$_POST['Aislamiento_acometida'];
		
			if($model->save())
				$this->redirect(array('/aislamiento_tierra/view?Toma='.$model->Toma));
		}

		$this->render('update',array(
			'model'=>$model,
		));
	}

	public function actionDelete()
	{
		if(Yii::app()->request->isPostRequest)
		{
                     $model= $this->loadModel();
                    $mitoma=$model->Toma;
			$this->loadModel()->delete();

			if(!isset($_GET['ajax']))
				$this->redirect(array('/aislamiento_tierra/view?Toma='.$mitoma));
		}
		else
			throw new CHttpException(400,
					Yii::t('app', 'Invalid request. Please do not repeat this request again.'));
	}

	public function actionIndex()
	{
		$dataProvider=new CActiveDataProvider('Aislamiento_acometida');
		$this->render('index',array(
			'dataProvider'=>$dataProvider,
		));
	}

	public function actionAdmin()
	{
		$model=new Aislamiento_acometida('search');
		if(isset($_GET['Aislamiento_acometida']))
			$model->attributes=$_GET['Aislamiento_acometida'];

		$this->render('admin',array(
			'model'=>$model,
		));
	}

	public function loadModel()
	{
		if($this->_model===null)
		{
			if(isset($_GET['id']))
				$this->_model=Aislamiento_acometida::model()->findbyPk($_GET['id']);
			if($this->_model===null)
				throw new CHttpException(404, Yii::t('app', 'The requested page does not exist.'));
		}
		return $this->_model;
	}

	protected function performAjaxValidation($model)
	{
		if(isset($_POST['ajax']) && $_POST['ajax']==='aislamiento-acometida-form')
		{
			echo CActiveForm::validate($model);
			Yii::app()->end();
		}
	}
}
