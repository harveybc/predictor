<?php
$this->breadcrumbs=array(
	'Informes de Tableros Eléctricos'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Informes', 'url'=>array('index')),
	array('label'=>'Nuevo Informe', 'url'=>array('create')),
	array('label'=>'Ver Informe', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Informes', 'url'=>array('admin')),
);
?>

<?php
$nombre="";
$modelTMP=$model;
if (isset($modelTMP->TAG))
        $nombre=$modelTMP->TAG;
if (isset($modelTMP->Fecha))
        $nombre=$nombre.' ('.$modelTMP->Fecha.')';
?>

<?php $this->setPageTitle (' Actualizar Informe de:'.$nombre.''); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'termotablero-form',
	'enableAjaxValidation'=>true,
    'htmlOptions'=>array('enctype' => 'multipart/form-data'),
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
    'modelArchivo' => $modelArchivo,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Actualizar')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
