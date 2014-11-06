<?php
$this->breadcrumbs=array(
	'Préstamos'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Préstamos', 'url'=>array('index')),
	array('label'=>'Nuevo Préstamo', 'url'=>array('create')),
	array('label'=>'Detalles Préstamo', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Préstamos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Préstamo #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'prestamos-form',
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
