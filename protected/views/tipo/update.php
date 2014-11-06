
<?php
    //para leer el param get y con el reconfigurar los dropdown
    $valor="";
    if  (isset($_GET['id']))
    {
        $valor=$_GET['id'];
    }
?>

<?php
$this->breadcrumbs=array(
	'Lubricantes'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Lubricantes', 'url'=>array('index')),
	array('label'=>'Nuevo Lubricante', 'url'=>array('create')),
	array('label'=>'Detalles Lubricantes', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Lubricantes', 'url'=>array('admin')),
);
?>

<?php
$nombre="";
$modelTMP=$model;
if (isset($modelTMP->Tipo_Aceite))
        $nombre=$modelTMP->Tipo_Aceite;
?>

<?php $this->setPageTitle(' Actualizar Lubricante:'.$nombre.''); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'tipo-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form,
        'valor'=>$valor
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Actualizar')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
