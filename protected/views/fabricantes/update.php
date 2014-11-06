<?php
$this->breadcrumbs=array(
	'Fabricantes'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Fabricantes', 'url'=>array('index')),
	array('label'=>'Nuevo Fabricante', 'url'=>array('create')),
	array('label'=>'Detalles Fabricante', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Fabricantes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Fabricantes #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'fabricantes-form',
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
