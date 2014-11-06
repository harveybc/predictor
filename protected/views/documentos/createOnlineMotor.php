

<?php
$this->breadcrumbs=array(
	'Documentos'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Documentos', 'url'=>array('index')),
	array('label'=>'Gestionar Documentos', 'url'=>array('admin')),
);
?>
<br/>
<br/>

<?php $this->setPageTitle (' Crear Documento Online de Motor'); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'documentos-form',
	'enableAjaxValidation'=>true,
        'htmlOptions'=>array('enctype' => 'multipart/form-data'),
)); 
echo $this->renderPartial('_formOnlineMotor', array(
	'model'=>$model,
        'modelMeta'=>$modelMeta,
        'modelA'=>$modelA,
	'form' =>$form
	)); ?>

<div class="row buttons">
	<?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
