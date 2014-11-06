<?php
$this->breadcrumbs=array(
	'Errores De Pegados'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Update'),
);

$this->menu=array(
	array('label'=>'List Errores_de_pegado', 'url'=>array('index')),
	array('label'=>'Create Errores_de_pegado', 'url'=>array('create')),
	array('label'=>'View Errores_de_pegado', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Manage Errores_de_pegado', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Update Errores_de_pegado #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'errores-de-pegado-form',
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
