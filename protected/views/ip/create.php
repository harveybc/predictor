<?php
$this->breadcrumbs=array(
	'Ips'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Ip', 'url'=>array('index')),
	array('label'=>'Gestionar Ip', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Nueva Ip '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'ip-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Guardar')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
