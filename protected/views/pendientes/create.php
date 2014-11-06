<?php
$this->breadcrumbs=array(
	'Pendientes'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Create'),
);

$this->menu=array(
	array('label'=>'List Pendientes', 'url'=>array('index')),
	array('label'=>'Manage Pendientes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Create Pendientes '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'pendientes-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Create')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
