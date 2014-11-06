<?php
$this->breadcrumbs=array(
	'Autorizaciones'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Autorizaciones', 'url'=>array('index')),
	array('label'=>'Nueva Autorización', 'url'=>array('create')),
	array('label'=>'Detalles Autorización', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Autorizaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Autorizaciones #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'autorizaciones-form',
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
