<?php
$this->breadcrumbs=array(
	'Errores De Pegados'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Create'),
);

$this->menu=array(
	array('label'=>'List Errores_de_pegado', 'url'=>array('index')),
	array('label'=>'Manage Errores_de_pegado', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Create Errores_de_pegado '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'errores-de-pegado-form',
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
