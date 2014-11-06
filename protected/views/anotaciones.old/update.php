<?php
$this->breadcrumbs=array(
	'Anotaciones'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Anotaciones', 'url'=>array('index')),
	array('label'=>'Nueva Anotación', 'url'=>array('create')),
	array('label'=>'Detalles Anotación', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Anotaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Anotaciones #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'anotaciones-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Actualizar')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
