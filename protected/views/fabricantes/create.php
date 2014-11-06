<?php
$this->breadcrumbs=array(
	'Fabricantes'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Fabricantes', 'url'=>array('index')),
	array('label'=>'Gestionar Fabricantes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Fabricantes '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'fabricantes-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
