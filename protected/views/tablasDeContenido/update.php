<?php
$this->breadcrumbs=array(
	'Tablas De Contenidos'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Update'),
);

$this->menu=array(
	array('label'=>'Lista de TablasDeContenido', 'url'=>array('index')),
	array('label'=>'Create TablasDeContenido', 'url'=>array('create')),
	array('label'=>'View TablasDeContenido', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Manage TablasDeContenido', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Update TablasDeContenido #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'tablas-de-contenido-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Update')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
