<?php
$this->breadcrumbs=array(
	'Permisos'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Permisos', 'url'=>array('index')),
	array('label'=>'Nuevo Permiso', 'url'=>array('create')),
	array('label'=>'Detalles Permiso', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Permisos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Permisos #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'permisos-form',
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
